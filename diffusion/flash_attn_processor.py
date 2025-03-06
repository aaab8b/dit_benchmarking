import torch
from diffusers.configuration_utils import register_to_config
from typing import Any, Dict, Optional, Tuple, Union
# from diffusers.models.embeddings import  CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings,FluxPosEmbed
import torch.nn as nn
# from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock,FluxTransformerBlock
# from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward
import torch.nn.functional as F
from flash_attn.modules.mha import FlashSelfAttention,FlashCrossAttention
from flash_attn import flash_attn_func
def set_attn_processor(model, processor):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        # count = len(model.attn_processors.keys())

        # if isinstance(processor, dict) and len(processor) != count:
        #     raise ValueError(
        #         f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
        #         f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
        #     )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    # print(name)
                    module.set_processor(processor)

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in model.named_children():
            fn_recursive_attn_processor(name, module, processor)


class JointFlashAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        self.inner_attn=FlashSelfAttention()
        # if not hasattr(F, "scaled_dot_product_attention"):
        #     raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        # all (b,h,s,d) with different sq and sk
        orig_datatype = query.dtype
        query = query.to(torch.bfloat16)
        key = key.to(torch.bfloat16)
        value = value.to(torch.bfloat16)
        qkv = torch.stack([query,key,value],dim=2).transpose(1,3)
        # qkv = torch.stack([query, key, value], dim=2).transpose(1,3)     # [b, s, 3, h, d]
        hidden_states = self.inner_attn(qkv)
        hidden_states = hidden_states.reshape(batch_size,-1,attn.heads * head_dim)
        hidden_states = hidden_states.to(orig_datatype)

        # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

class FlashAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        # if not hasattr(F, "scaled_dot_product_attention"):
        #     raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        # self.inner_attn=FlashSelfAttention()
        print("using flash attn func")
        self.inner_attn=flash_attn_func
        # torch.compiler.disable(self.inner_attn)
        # torch._dynamo.disallow_in_graph(self.inner_attn)
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )
        orig_datatype = query.dtype
        # query = query.to(torch.bfloat16)
        # key = key.to(torch.bfloat16)
        # value = value.to(torch.bfloat16)
        # qkv = torch.stack([query, key, value], dim=2).transpose(1,3)     # [b, s, 3, h, d]
        # hidden_states=self.inner_attn(qkv)
        query = query.to(torch.bfloat16).transpose(1,2)
        key = key.to(torch.bfloat16).transpose(1,2)
        value = value.to(torch.bfloat16).transpose(1,2)
        hidden_states = self.inner_attn(query,key,value,dropout_p=0,softmax_scale=None,causal=False,deterministic=False)
        hidden_states = hidden_states.reshape(batch_size,-1,attn.heads * head_dim)
        # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # hidden_states = hidden_states.to(query.dtype)
        hidden_states = hidden_states.to(orig_datatype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

# if __name__=="__main__":
#     from diffusers import FluxTransformer2DModel
#     FluxSingleTransformerBlock.__init__=flash_single_init
#     FluxTransformerBlock.__init__=flash_init
#     transformer = FluxTransformer2DModel.from_pretrained(
#                 'black-forest-labs/FLUX.1-dev', subfolder="transformer")
#     transformer = transformer.to(torch.float32)
#     transformer.train()
if __name__=="__main__":
    from diffusers import DiTTransformer2DModel
    # model=DiTTransformer2DModel(
    #     sample_size=32,
    #     num_embeds_ada_norm=64,
    #     in_channels=4,
    #     out_channels=8
    #     )
    from diffusers.models.attention import Attention
    processor=JointFlashAttnProcessor2_0()
    attn = Attention(
        query_dim=1536,
        cross_attention_dim=None,
        added_kv_proj_dim=1536,
        dim_head=64,
        heads=24,
        out_dim=1536,
        context_pre_only=False,
        bias=True,
        processor=processor,
        qk_norm=None,
        eps=1e-6,
    ).cuda()
    hidden_states=torch.randn([4,4096,1536]).cuda()
    encoder_hidden_states=torch.randn([4,154,1536]).cuda()
    result=attn(hidden_states,encoder_hidden_states)
    print(result[0].shape)
    print(result[1].shape)