import logging
import math
import os
import time
import torch.nn.functional as F
from sgEncoderTraining.training.precision import get_autocast
from sgEncoderTraining.global_var import *
from tqdm import tqdm


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate_by_iter(step,
                     model,
                     dataloader,
                     epoch,
                     args,
                     vae,
                     unet,
                     noise_scheduler,
                     accelerator,
                     tb_writer=None):

    model.eval()

    autocast = get_autocast(args.precision)

    batch_size = args.val_batch_size

    total_loss_m = AverageMeter()

    if accelerator.is_main_process:
        dataloader = tqdm(dataloader, desc=f"VAL Epoch {epoch + 1}/{args.epochs}_{step}")

    for i, batch in enumerate(dataloader):

        all_imgs, all_triples, all_global_ids, all_isolated_items, all_text_prompts, all_original_sizes, all_crop_top_lefts = [
            x for x in batch]

        all_imgs = torch.cat(all_imgs, dim=0).to(accelerator.device)

        model_input = vae.encode(all_imgs).latent_dist.sample()
        model_input = model_input * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input)

        bsz = model_input.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device
        )
        timesteps = timesteps.long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

        # time ids
        def compute_time_ids(original_size, crops_coords_top_left):
            # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
            target_size = (args.image_size, args.image_size)
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids])
            add_time_ids = add_time_ids.to(accelerator.device, dtype=torch.float16)
            return add_time_ids

        add_time_ids = torch.cat(
            [compute_time_ids(s, c) for s, c in zip(all_original_sizes, all_crop_top_lefts)]
        )

        unet_added_conditions = {"time_ids": add_time_ids}

        with autocast():
            prompt_embeds, pooled_embeds = model(all_triples, all_isolated_items, all_global_ids)
            unet_added_conditions.update({"text_embeds": pooled_embeds})

            model_pred = unet(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                added_cond_kwargs=unet_added_conditions,
                return_dict=False,
            )[0]

            target = noise

            val_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        if accelerator.is_main_process:
            total_loss_m.update(val_loss.item(), batch_size)

    if accelerator.is_main_process:

        logging.info(
            f"Validate Epoch: {epoch} "
            f"Validate Iteration: {step} "
            f"Validate Batch Size: {batch_size} "
            f"Average mse loss of ori_img2sdxl_img: {total_loss_m.avg:#.5g} "
        )

        log_data = {
            "val_loss": total_loss_m.avg,
        }
        for name, val in log_data.items():
            name = "validate/" + name
            if tb_writer is not None:
                tb_writer.add_scalar(name, val, step)

def train_by_iters(model,
                   dataloader,
                   val_dataloader,
                   epoch,
                   optimizer,
                   scaler,
                   scheduler,
                   args,
                   vae,
                   unet,
                   noise_scheduler,
                   accelerator,
                   tb_writer=None,
                   val_count=10,
                   accumulation_steps=2):

    autocast = get_autocast(args.precision)

    model.train()

    unet.train()

    accumulation_steps = args.accusteps if hasattr(args, 'accusteps') else accumulation_steps

    optimizer.zero_grad()

    num_batches_per_epoch = len(dataloader)

    val_frequency = num_batches_per_epoch // val_count  # 每val_count%的iter进行一次验证

    sample_digits = math.ceil(math.log(num_batches_per_epoch + 1, 10))

    total_loss_m = AverageMeter()
    batch_time_m = AverageMeter()

    data_time_m = AverageMeter()
    end = time.time()

    if accelerator.is_main_process:
        dataloader = tqdm(dataloader, total=num_batches_per_epoch, desc=f"Epoch {epoch + 1}/{args.epochs}")

    dataloader = tqdm(dataloader, total=num_batches_per_epoch, desc=f"Epoch {epoch + 1}/{args.epochs}")

    for i, batch in enumerate(dataloader):

        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        all_imgs, all_triples, all_global_ids, all_isolated_items, all_text_prompts, all_original_sizes, all_crop_top_lefts, all_img_ids = [
            x for x in batch]

        all_imgs = torch.cat(all_imgs, dim=0).to(accelerator.device)

        data_time_m.update(time.time() - end)

        model_input = vae.encode(all_imgs).latent_dist.sample()
        model_input = model_input * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input)

        bsz = model_input.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device
        )
        timesteps = timesteps.long()

        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

        # time ids
        def compute_time_ids(original_size, crops_coords_top_left):
            # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
            target_size = (args.image_size, args.image_size)
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids])
            add_time_ids = add_time_ids.to(accelerator.device, dtype=torch.float16)
            return add_time_ids

        add_time_ids = torch.cat(
            [compute_time_ids(s, c) for s, c in zip(all_original_sizes, all_crop_top_lefts)])

        unet_added_conditions = {"time_ids": add_time_ids}

        with autocast():
            # applying sg
            prompt_embeds, pooled_embeds = model(all_triples, all_isolated_items, all_global_ids)
            unet_added_conditions.update({"text_embeds": pooled_embeds})

            model_pred = unet(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                added_cond_kwargs=unet_added_conditions,
                return_dict=False,
            )[0]

            target = noise

            total_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")


        if scaler is not None:

            accelerator.backward(scaler.scale(total_loss))

            if args.norm_gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            accelerator.backward(total_loss)

            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        total_loss_m.update(total_loss.item(), accumulation_steps)

        if accelerator.is_main_process and ((i + 1) % accumulation_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = accumulation_steps
            num_samples = batch_count * batch_size
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            total_loss_m.update(total_loss.item(), batch_size)
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}} ({percent_complete:.0f}%)] "
                f"Total Loss: {total_loss_m.val:#.5g} ({total_loss_m.avg:#.4g}) "

                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
            )

            log_data = {
                "batch_loss": total_loss_m.val,

                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size / batch_time_m.val,

                "lr": optimizer.param_groups[0]["lr"],
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)

            batch_time_m.reset()
            data_time_m.reset()

        if (step + 1) % val_frequency == 0 or (step + 1) % num_batches_per_epoch == 0:
            if accelerator.is_main_process:

                if tb_writer is not None:
                    avg_loss = total_loss_m.avg
                    tb_writer.add_scalar("iter_avg_loss", avg_loss, step)
                    tb_writer.flush()

                    logging.info(f"Average Loss for Epoch {epoch} Iter {step}: {avg_loss:.4f}")

                completed_epoch = epoch + 1

                if args.save_logs:
                    checkpoint_dict = {
                        "epoch": completed_epoch,
                        "iteration": step,
                        "name": args.name,
                        "state_dict": accelerator.get_state_dict(model),
                        "optimizer": optimizer.state_dict(),
                    }
                    if scaler is not None:
                        checkpoint_dict["scaler"] = scaler.state_dict()

                    if completed_epoch == args.epochs or (
                            args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
                    ):
                        torch.save(
                            checkpoint_dict,
                            os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}_iter_{step}.pt"), )

            with torch.no_grad():
                    validate_by_iter(step, model, val_dataloader, epoch, args, vae,
                                 unet, noise_scheduler, accelerator, tb_writer)





