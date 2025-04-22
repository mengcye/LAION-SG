from diffusers import StableDiffusionXLPipeline
import torch.utils.checkpoint
from sgEncoderTraining.datasets.laion_dataset import build_laion_loaders
from configs.configs_laion import parse_args
from transformers import AutoTokenizer, PretrainedConfig
from sgEncoderTraining.sgEncoder.create_sg_encoder import create_model_and_transforms

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)

args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.val_batch_size = 1
args.batch_size = 1

train_dataloader, val_dataloader = build_laion_loaders(args=args)

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

checkpoint = torch.load("./baseline3_100.pt", map_location=device)

model.load_state_dict(checkpoint['state_dict'])


if __name__ == '__main__':

    real_images = []
    fake_images = []
    args.num_inference_steps = 40
    size = 1024

    for i, batch in enumerate(val_dataloader):

        all_imgs, all_triples, all_global_ids, all_isolated_items, all_text_prompts, all_original_sizes, all_crop_top_lefts, all_ids = batch

        label = all_text_prompts[0]

        real_images.extend(all_imgs)

        prompt_embeds, pooled_embeds = model(all_triples, all_isolated_items, all_global_ids)

        img = pipeline(
            prompt_embeds=prompt_embeds,
            num_inference_steps=args.num_inference_steps,
            pooled_prompt_embeds=pooled_embeds,
            width=size,
            height=size,
        ).images[0]

        # Specify the file path and save the image
        file_path = f"./output/{all_ids[0]}.jpg"  # Update the path accordingly
        img.save(file_path, format='JPEG')
