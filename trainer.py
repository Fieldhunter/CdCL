import os
import utility
import torch
from decimal import Decimal
from loss.infoNCE import InfoNCE
from utils import util
import math
import warnings
warnings.filterwarnings("ignore")


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        if args.n_GPUs > 1:
            self.model_E = torch.nn.parallel.DistributedDataParallel(self.model.get_model().E, [loader.local_rank],
                                                                     find_unused_parameters=True)
        else:
            self.model_E = self.model.get_model().E
        self.loss = my_loss
        self.contrast_loss = torch.nn.CrossEntropyLoss().cuda()
        self.criterion = InfoNCE(1.0, args.n_GPUs).cuda()
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if args.n_GPUs > 1:
            if self.args.resume != 0 and not self.args.test_only:
                self.optimizer.load_state_dict(
                    torch.load(os.path.join(self.ckp.dir, 'opt', 'opt_{}.pt'.format(self.args.resume)),
                               map_location='cuda:{}'.format(args.local_rank))
                )
                for _ in range(self.args.resume): self.scheduler.step()
        else:
            if self.args.resume != 0 and not self.args.test_only:
                self.optimizer.load_state_dict(
                    torch.load(os.path.join(self.ckp.dir, 'opt', 'opt_{}.pt'.format(self.args.resume)))
                )
                for _ in range(self.args.resume): self.scheduler.step()

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1

        # lr stepwise
        if epoch <= self.args.epochs_encoder:
            lr = self.args.lr_encoder * (self.args.gamma_encoder ** (epoch // self.args.lr_decay_encoder))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            if epoch > self.args.epochs_encoder and epoch <= (self.args.epochs_encoder + 5):
                lr = self.args.max_lr_sr * (epoch - self.args.epochs_encoder) / 5
            else:
                lr = self.args.min_lr_sr + (self.args.max_lr_sr - self.args.min_lr_sr) * \
                     (1 + math.cos((epoch - (self.args.epochs_encoder + 5)) / (self.args.epochs_sr - 5) * math.pi)) / 2
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()

        # TODO: setting 1 or 2
        degrade = util.SRMDPreprocessing(
            self.scale[0],
            kernel_size=self.args.blur_kernel,
            blur_type=self.args.blur_type,
            sig_min=self.args.sig_min,
            sig_max=self.args.sig_max,
            lambda_min=self.args.lambda_min,
            lambda_max=self.args.lambda_max,
            noise=self.args.noise
        )

        # TODO: setting 3
        # degrade = util.complex_SRMDPreprocessing(
        #     self.scale[0],
        #     kernel_size=self.args.blur_kernel,
        #     blur_type=self.args.blur_type,
        #     sig_min=self.args.sig_min,
        #     sig_max=self.args.sig_max,
        #     lambda_min=self.args.lambda_min,
        #     lambda_max=self.args.lambda_max,
        #     noise=self.args.noise,
        #     noise_min=self.args.noise_min,
        #     noise_max=self.args.noise_max,
        #     jpeg_min=self.args.jpeg_min,
        #     jpeg_max=self.args.jpeg_max,
        #     istrain=True
        # )

        timer = utility.timer()
        losses_contrast, losses_sr = utility.AverageMeter(), utility.AverageMeter()

        for batch, (hr, _, idx_scale) in enumerate(self.loader_train):
            hr = hr.cuda()                              # b, n, c, h, w

            hr = utility.rotate(hr, 4)  # b, n, c, h, w

            lr = degrade(hr)                 # b, n, c, h, w
            self.optimizer.zero_grad()

            timer.tic()
            # forward
            ## train degradation encoder
            if epoch <= self.args.epochs_encoder:
                _, all_p, all_z = self.model_E(lr)  # fast-mocov3 + multi-crop
                loss_constrast = self.criterion.forward(all_p, all_z)  # fast-mocov3 + multi-crop
                loss = loss_constrast

                losses_contrast.update(loss_constrast.item())
            ## train the whole network
            else:
                sr, all_p, all_z = self.model(lr)  # fast-mocov3 + multi-crop
                loss_SR = self.loss(sr*255., hr[:, 0, ...])
                loss_constrast = self.criterion.forward(all_p, all_z)  # fast-mocov3 + multi-crop
                loss = loss_constrast + loss_SR

                losses_sr.update(loss_SR.item())
                losses_contrast.update(loss_constrast.item())

            # backward
            loss.backward()
            self.optimizer.step()
            timer.hold()

            if epoch <= self.args.epochs_encoder:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:03d}][{:04d}/{:04d}]\t'
                        'Loss [contrastive loss: {:.3f}]\t'
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_contrast.avg,
                            timer.release()
                        ))
            else:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:04d}][{:04d}/{:04d}]\t'
                        'Loss [SR loss:{:.3f} | contrastive loss: {:.3f}]\t'
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_sr.avg, losses_contrast.avg,
                            timer.release(),
                        ))

        self.loss.end_log(len(self.loader_train))

        # save model
        target = self.model.get_model()
        model_dict = target.state_dict()
        torch.save(
            model_dict,
            os.path.join(self.ckp.dir, 'model', 'model_{}.pt'.format(epoch))
        )
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(self.ckp.dir, 'opt', 'opt_{}.pt'.format(epoch))
        )

    def test(self):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()

        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                self.loader_test.dataset.set_scale(idx_scale)
                eval_psnr = 0
                eval_ssim = 0

                # TODO: setting 1 or 2
                degrade = util.SRMDPreprocessing(
                    self.scale[0],
                    kernel_size=self.args.blur_kernel,
                    blur_type=self.args.blur_type,
                    sig=self.args.sig,
                    lambda_1=self.args.lambda_1,
                    lambda_2=self.args.lambda_2,
                    theta=self.args.theta,
                    noise=self.args.noise
                )

                # TODO: setting 3
                # degrade = util.complex_SRMDPreprocessing(
                #     self.scale[0],
                #     kernel_size=self.args.blur_kernel,
                #     blur_type=self.args.blur_type,
                #     sig=self.args.sig,
                #     lambda_1=self.args.lambda_1,
                #     lambda_2=self.args.lambda_2,
                #     theta=self.args.theta,
                #     noise=self.args.noise,
                #     jpeg=self.args.jpeg,
                #     istrain=False
                # )

                for idx_img, (hr, filename, _) in enumerate(self.loader_test):
                    hr = hr.cuda()                      # b, 1, c, h, w
                    hr = self.crop_border(hr, scale)

                    # TODO: setting 1 or 2
                    lr = degrade(hr, random=False)[:, 0, ...]
                    hr = hr[:, 0, ...]
                    # TODO: setting 3
                    # hr = hr[:, 0, ...]
                    # lr = degrade(hr, random=False)

                    # inference
                    timer_test.tic()
                    sr = self.model(lr) * 255.
                    timer_test.hold()

                    sr = utility.quantize(sr, self.args.rgb_range)
                    hr = utility.quantize(hr, self.args.rgb_range)

                    # metrics
                    eval_psnr += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    eval_ssim += utility.calc_ssim(
                        sr, hr, scale,
                        benchmark=self.loader_test.dataset.benchmark
                    )

                    # save results
                    if self.args.save_results:
                        save_list = [sr]
                        filename = filename[0]
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_psnr / len(self.loader_test)
                self.ckp.write_log(
                    '[Epoch {}---{} x{}]\tPSNR: {:.3f} SSIM: {:.4f}'.format(
                        self.args.resume,
                        self.args.data_test,
                        scale,
                        eval_psnr / len(self.loader_test),
                        eval_ssim / len(self.loader_test),
                    ))

    def crop_border(self, img_hr, scale):
        b, n, c, h, w = img_hr.size()

        img_hr = img_hr[:, :, :, :int(h//scale*scale), :int(w//scale*scale)]

        return img_hr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs_encoder + self.args.epochs_sr


