import torch
import torch.nn.functional as F
import torch.distributed as dist
from utils import dist as link
from torch.nn.modules.loss import _Loss


class InfoNCE(_Loss):
    def __init__(self, temperature, n_gpu):
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        self.n_gpu = n_gpu

    @staticmethod
    def cosine_similarity(p, z):
        # [N, E]

        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)
        # [N E] [N E] -> [N] -> [1]
        return (p * z).sum(dim=1).mean()  # dot product & batch coeff normalization

    def loss(self, p, z_gather):
        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)

        offset = link.get_rank() * p.shape[1]
        labels = torch.arange(offset, offset + p.shape[1], dtype=torch.long).cuda()
        p_z_m = p.bmm(torch.t(z_gather).repeat(p.shape[0], 1, 1)) / self.temperature  # [N_local, N]

        return sum(map(lambda x: F.cross_entropy(x, labels), p_z_m))

    def forward(self, all_p, all_z):  # multi-crop version
        batch_size = all_z[0].size(0)
        all_p = list(map(lambda x: torch.stack(x.split(batch_size, dim=0), dim=0), all_p))
        if self.n_gpu > 1:
            all_z = list(map(lambda x: concat_all_gather((x / x.norm(dim=-1, keepdim=True)).detach()), all_z))
        else:
            all_z = list(map(lambda x: (x / x.norm(dim=-1, keepdim=True).detach()), all_z))

        loss = 0
        for i in range(len(all_p)):
            for j in range(len(all_z)):
                if i == j:
                    continue
                loss += self.loss(all_p[i], all_z[j])

        return loss / (len(all_p) * (len(all_z) - 1) * len(all_p[0]))


@torch.no_grad()
def concat_all_gather(tensor):
    """gather the given tensor"""
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
