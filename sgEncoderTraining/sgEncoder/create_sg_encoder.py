import torch
import json
from sgEncoderTraining.sgEncoder.sg_encoder import sgEncoder, convert_weights_to_fp16

def create_model(
        args,
        text_encoders: list,
        tokenizers: list,
        model_config_json: str,
        precision: str = 'fp32',
        device: torch.device = torch.device('cpu'),
        force_quick_gelu: bool = False,
        pretrained_image: bool = False,
):
    if model_config_json != '':
        with open(model_config_json, 'r') as f:
            model_cfg = json.load(f)
    else:
        model_cfg = {
            "graph_cfg": {
                "layers": args.num_graph_layer,
                "width": args.graph_width,
            },
            "embed_dim": args.embed_dim,
        }

    if force_quick_gelu:
        model_cfg["quick_gelu"] = True

    if pretrained_image:
        if 'timm_model_name' in model_cfg.get('vision_cfg', {}):
            model_cfg['vision_cfg']['timm_model_pretrained'] = True
        else:
            assert False, 'pretrained image towers currently only supported for timm models'

    model = sgEncoder(text_encoders=text_encoders, tokenizers=tokenizers, **model_cfg)

    model.to(device=device)
    if precision == "fp16":
        assert device.type != 'cpu'
        convert_weights_to_fp16(model)

    return model

def create_model_and_transforms(
        args,
        text_encoders: list,
        tokenizers: list,
        model_config_json: str,
        precision: str = 'fp32',
        device: torch.device = torch.device('cpu'),
        force_quick_gelu: bool = False,
        pretrained_image: bool = False,
):
    model = create_model(args, text_encoders,tokenizers, model_config_json, precision, device, force_quick_gelu=force_quick_gelu, pretrained_image=pretrained_image)

    return model