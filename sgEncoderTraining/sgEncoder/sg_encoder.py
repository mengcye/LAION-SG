import numpy as np
import torch.nn.functional as F
from torch import nn
from sgEncoderTraining.sgEncoder.module import GraphTripleConv, GraphTripleConvNet, Attention
from configs.configs_laion import CLIPGraphCfg
from sgEncoderTraining.global_var import *

class sgEncoder(nn.Module):
    def __init__(self,
                 graph_cfg: CLIPGraphCfg,
                 text_encoders: list,
                 tokenizers: list,
                 embed_dim=512,
                 max_sample_per_img: int = 15,
                 clip_dim=2048,
                 ):
        super().__init__()
        if isinstance(graph_cfg, dict):
            graph_cfg = CLIPGraphCfg(**graph_cfg)

        self.max_sample_per_img = max_sample_per_img

        self.graph_conv = GraphTripleConv(embed_dim, output_dim=embed_dim, hidden_dim=graph_cfg.width, pooling='avg', mlp_normalization='none')
        self.graph_net = GraphTripleConvNet(embed_dim, num_layers=graph_cfg.layers, hidden_dim=graph_cfg.width, pooling='avg', mlp_normalization='none')

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.alpha_obj = nn.Parameter(torch.zeros(1))

        self.tokenizers = tokenizers

        self.text_encoders = text_encoders

        self.projection_mlp_1 = nn.Linear(clip_dim, embed_dim,bias=False)
        self.projection_mlp_2 = nn.Linear(embed_dim, clip_dim,bias=False)

    def initialize_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

        nn.init.constant_(self.projection_mlp_1.weight, 0.0)
        nn.init.constant_(self.projection_mlp_2.weight, 0.0)

        if hasattr(self.graph_conv, 'init_parameters'):
            self.graph_conv.init_parameters()
        if hasattr(self.graph_net, 'init_parameters'):
            self.graph_net.init_parameters()

        self.alpha_obj.data.fill_(0.0)

    def get_triple_embeddings(self, triples, isolated_items):
        sos_embeddings = []
        eos_embeddings = []

        pooled_embeddings = []

        obj_embeddings = []
        pred_embeddings = []
        edges = []

        index = 0

        triple_o_p_o_index_list=[]

        attri_embedding,_ = self.encode_prompt("-")
        attri_embedding = attri_embedding[:, 1:-1, :]

        device = attri_embedding.device

        for triple in triples:
            s_embedding, pooled_s_embedding = self.encode_prompt(str(triple['item1']))  #  (1, n, 768)
            p_embedding, pooled_p_embedding = self.encode_prompt(str(triple['relation']))
            o_embedding, pooled_o_embedding = self.encode_prompt(str(triple['item2']))

            sos_embeddings.append(s_embedding[:, 0, :]) # sos of s
            eos_embeddings.append(s_embedding[:, -1, :]) #eos of s
            pooled_embeddings.append(pooled_s_embedding)

            #obj and attribute of s
            slice = s_embedding[:,1:-1,:]
            s_len = slice.shape[1]
            obj_embeddings.append(slice)

            attri_counts = s_len - 1

            for _ in range(attri_counts):
                pred_embeddings.append(attri_embedding)

            s_index_in_obj = index + s_len - 1
            for i in range(attri_counts):
                edges.append([i + index,s_index_in_obj])
            index += s_len


            sos_embeddings.append(o_embedding[:, 0, :])
            eos_embeddings.append(o_embedding[:, -1, :])
            pooled_embeddings.append(pooled_o_embedding)


            slice = o_embedding[:,1:-1,:]
            o_len = slice.shape[1]
            obj_embeddings.append(slice)

            attri_counts = o_len - 1

            for _ in range(attri_counts):
                pred_embeddings.append(attri_embedding)


            o_index_in_obj = index + o_len - 1
            for i in range(attri_counts):
                edges.append([i + index, o_index_in_obj])
            index += o_len

            p_start = sum(tensor.shape[1] for tensor in pred_embeddings)

            sos_embeddings.append(p_embedding[:, 0, :])  # 拿 p 的 sos
            eos_embeddings.append(p_embedding[:, -1, :])  # 拿 p 的 eos
            pooled_embeddings.append(pooled_p_embedding)


            p_self = p_embedding[:, 1:-1, :]

            size_p = p_self.shape[1]
            pred_embeddings.append(p_self)

            for i in range(size_p):
                edges.append([s_index_in_obj, o_index_in_obj])

            p_location = [p_start, size_p + p_start - 1]

            triple_o_p_o_index_list.append([s_index_in_obj, p_location, o_index_in_obj])

        isolated_embeddings = []
        for item in isolated_items:
            item_embedding, pooled_item_embedding = self.encode_prompt(item)  #  (1, n, 768)

            sos_embeddings.append(item_embedding[:, 0, :])
            eos_embeddings.append(item_embedding[:, -1, :])
            pooled_embeddings.append(pooled_item_embedding)

            isolated_embeddings.append(item_embedding[:, 1:-1, :]) #(1, n-2, 768)

        edges = torch.tensor(edges, device=device)

        return sos_embeddings, eos_embeddings,pooled_embeddings, isolated_embeddings,obj_embeddings,pred_embeddings,edges,triple_o_p_o_index_list

    def get_text_embeddings(self, triples, isolated_items):

        triple_str = ''

        for triple in triples:
            triple_str += str(triple['item1']) + ' ' + str(triple['relation']) + ' ' + str(triple['item2']) + ' '

        for item in isolated_items:
            triple_str += item + ' '

        triple_embedding, pooled_triple_embedding = self.encode_prompt(triple_str)  #  (1, n, 768)
        sos_embedding = triple_embedding[:, 0, :]  # (1, 768)
        eos_embedding = triple_embedding[:, -1, :]  #  (1, 768)
        triple_embedding = triple_embedding[:, 1:-1, :]  # (1, n-2, 768)

        return sos_embedding, triple_embedding, eos_embedding, pooled_triple_embedding

    def generate_total_embedding_for_unet(self,
                                          be_prompt_embeddings,
                                          sos_embeddings,
                                          eos_embeddings,
                                          pooled_embeddings,
                                          obj_embeddings,
                                          pred_embeddings,
                                          triple_o_p_o_index_list):

        prompt_embeddings = []

        o_begin_index = 0
        for indexs in triple_o_p_o_index_list:
            o_index,p_index,s_index = indexs
            prompt_embeddings.append(obj_embeddings[o_begin_index:o_index + 1])
            prompt_embeddings.append(pred_embeddings[p_index[0]:p_index[1] + 1])
            prompt_embeddings.append(obj_embeddings[o_index + 1:s_index + 1])

            o_begin_index = s_index + 1

        prompt_embeddings = torch.cat(prompt_embeddings, dim=0).unsqueeze(0)

        clamped_alpha_o = self.alpha_obj

        c = prompt_embeddings.size(1)
        if c > 75:
            c = 75
            prompt_embeddings = prompt_embeddings[:, :c, :]

        be_prompt_embeddings[:, :c, :] += clamped_alpha_o * prompt_embeddings

        final_embeddings = torch.cat(
            (sos_embeddings.unsqueeze(0), be_prompt_embeddings, eos_embeddings.unsqueeze(0)), dim=1)

        target_length = 77
        current_length = final_embeddings.shape[1]

        if current_length < target_length:
            pad_length = target_length - current_length
            pad_embeddings = eos_embeddings.repeat(1, pad_length, 1)  #  (1, pad_length, 768)
            final_embeddings = torch.cat((final_embeddings, pad_embeddings), dim=1)
        elif current_length > target_length:
            final_embeddings = final_embeddings[:, :target_length, :]

        return final_embeddings, pooled_embeddings


    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=False):
        self.visual.set_grad_checkpointing(enable)

    def do_avg_pool(self,embeddings, adaptive_avg_pool):
        # 对 embeddings 执行池化操作
        embeddings = embeddings.unsqueeze(0)  # 将形状变为 (1, a, 768)
        embeddings_pooled = adaptive_avg_pool(embeddings.permute(0, 2, 1)).permute(0, 2, 1)  # (1, 1, 768)
        return embeddings_pooled

    def tokenize_prompt(self,tokenizer, prompt):
        text_inputs = tokenizer(
            prompt,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        return text_input_ids

    def encode_prompt(self, prompt):
        prompt_embeds_list = []

        for i, text_encoder in enumerate(self.text_encoders):
            tokenizer = self.tokenizers[i]
            text_input_ids = self.tokenize_prompt(tokenizer, prompt)

            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    def encode_graph_local_global(self, triples_list, isolated_items_list, item_list_list):

        final_embeddings_list = []
        pooled_embeddings_list = []
        for (triples, isolated_items, item_list) in zip(triples_list, isolated_items_list, item_list_list):

            (_, _, _, isolated_embeddings, obj_embeddings,
             pred_embeddings, edges, triple_o_p_o_index_list) = self.get_triple_embeddings(triples, isolated_items)

            (be_sos_embeddings, be_prompt_embeddings, be_eos_embeddings,
             be_pooled_embeddings) = self.get_text_embeddings(triples, isolated_items)

            obj_embeddings = torch.cat(obj_embeddings, dim=1).squeeze(0)
            pred_embeddings = torch.cat(pred_embeddings, dim=1).squeeze(0)

            obj_vecs = self.projection_mlp_1(obj_embeddings)
            pred_vecs = self.projection_mlp_1(pred_embeddings)

            obj_vecs, pred_vecs = self.graph_conv(obj_vecs, pred_vecs, edges)

            if self.graph_net is not None:
                obj_vecs, pred_vecs = self.graph_net(obj_vecs, pred_vecs, edges)

            obj_vecs = F.normalize(obj_vecs, p=2, dim=1)
            pred_vecs = F.normalize(pred_vecs, p=2, dim=1)

            obj_vecs = self.projection_mlp_2(obj_vecs)
            pred_vecs = self.projection_mlp_2(pred_vecs)

            final_embeddings, pooled_embeddings = self.generate_total_embedding_for_unet(be_prompt_embeddings,
                                                                                         be_sos_embeddings,
                                                                                         be_eos_embeddings,
                                                                                         be_pooled_embeddings,
                                                                                         obj_vecs,
                                                                                         pred_vecs,
                                                                                         triple_o_p_o_index_list)

            final_embeddings_list.append(final_embeddings)
            pooled_embeddings_list.append(pooled_embeddings)

        final_embeddings_all = torch.cat(final_embeddings_list, dim=0)
        pooled_embeddings_all = torch.cat(pooled_embeddings_list, dim=0)

        return final_embeddings_all, pooled_embeddings_all

    def forward(self, triples_list, isolated_items_list, item_list_list):

        final_embeddings_all, pooled_embeddings_all = self.encode_graph_local_global(triples_list, isolated_items_list, item_list_list)

        return  final_embeddings_all,pooled_embeddings_all

def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

