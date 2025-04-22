import os
import torch
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel
import torch.utils.checkpoint
from sgEncoderTraining.datasets.laion_dataset import build_laion_loaders
from configs.configs_laion import parse_args
from transformers import AutoTokenizer, PretrainedConfig
from sgEncoderTraining.sgEncoder.create_sg_encoder import create_model_and_transforms

accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

args = parse_args()

device = accelerator.device

args.image_size = 1024
args.val_batch_size = 1
args.batch_size = 1

args.val_json_path = f'put your val json file here'

_,val_loader = build_laion_loaders(args=args)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,  subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


vae = AutoencoderKL.from_pretrained(
        args.stable_diffusion_checkpoint,
        subfolder="vae",
        variant="fp16",
        cache_dir=args.cache_dir
    ).to(device)

tokenizer_one = AutoTokenizer.from_pretrained(
        args.stable_diffusion_checkpoint,
        subfolder="tokenizer",
        use_fast=False,
        cache_dir=args.cache_dir
    )

tokenizer_two = AutoTokenizer.from_pretrained(
        args.stable_diffusion_checkpoint,
        subfolder="tokenizer_2",
        use_fast=False,
        cache_dir=args.cache_dir
    )

text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.stable_diffusion_checkpoint
    )

text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.stable_diffusion_checkpoint, subfolder="text_encoder_2"
    )


text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.stable_diffusion_checkpoint, subfolder="text_encoder", variant="fp16",cache_dir=args.cache_dir).to(device)
text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.stable_diffusion_checkpoint, subfolder="text_encoder_2", variant="fp16",cache_dir=args.cache_dir).to(device)

unet = UNet2DConditionModel.from_pretrained(
    args.stable_diffusion_checkpoint, subfolder="unet", variant="fp16", cache_dir=args.cache_dir).to(device)


pipeline = StableDiffusionXLPipeline.from_pretrained(
    args.stable_diffusion_checkpoint,
    vae=vae,
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    unet=unet,
    torch_dtype='fp16',
    cache_dir=args.cache_dir
)

model = create_model_and_transforms(
        args,
        text_encoders=[text_encoder_one, text_encoder_two],
        tokenizers =[tokenizer_one,tokenizer_two],
        model_config_json=args.model_config_json,
        precision=args.precision,
        device=device,
        force_quick_gelu=args.force_quick_gelu,
        pretrained_image=args.pretrained_image,
    ).to(device)


checkpoint = torch.load("put your sg_encoder ckpt here", map_location=device)

model.load_state_dict(checkpoint['state_dict'])

model, val_loader, pipeline = accelerator.prepare(model, val_loader, pipeline)

pipeline.set_progress_bar_config(disable=True)

if __name__ == '__main__':
    
    npz_dir = f"npz_files_{accelerator.process_index}"
    os.makedirs(npz_dir, exist_ok=True)

    args.num_inference_steps = 40

    for i, batch in enumerate(tqdm(val_loader, desc="Processing Batches", unit="batch")):

        all_imgs, all_triples, all_global_ids, all_isolated_items, _, _, _,all_ids = batch

        prompt_embeds, pooled_embeds = model(all_triples, all_isolated_items, all_global_ids)

        img = pipeline(
            prompt_embeds=prompt_embeds,
            num_inference_steps=args.num_inference_steps,
            pooled_prompt_embeds=pooled_embeds,
            width=args.image_size,
            height=args.image_size,
            output_type="np"
        ).images

        real_images_tensor = torch.stack(all_imgs)
        real_images_np = real_images_tensor.squeeze(1).cpu().numpy()
        real_images_np = real_images_np.transpose(0, 2, 3, 1)
        real_images_np = (real_images_np * 255).astype(np.uint8)

        fake_images_np = (img * 255).astype(np.uint8)

        npz_file_path = os.path.join(npz_dir, f"batch_{i}.npz")
        np.savez_compressed(npz_file_path, real_images=real_images_np, fake_images=fake_images_np)

        del all_imgs, img, prompt_embeds, pooled_embeds, real_images_np, fake_images_np
        torch.cuda.empty_cache()
