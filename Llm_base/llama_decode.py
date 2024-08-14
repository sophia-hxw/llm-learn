import torch
from torch import nn
from transformers import LlamaConfig

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config : LlamaConfig):
        super.__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(hidden_size=self.hidden_size, intermediate=config.intermediate_size, hidden_act=config.hidden_act)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,
        hidden_status: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        )->Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residule = hidden_status
        hidden_status = self.input_layernorm(hidden_status)

        # self attention 
        hidden_status, self_attn_weights, present_key_value = self.self_attn(
            hidden_status=hidden_status,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_status = residule + hidden_status

        # fully connected
        residule = hidden_status
        hidden_status = self.post_attention_layernorm(hidden_status)
        hidden_status = self.mlp(hidden_status)
        hidden_status = residule + hidden_status
        
        outputs = (hidden_status, )
        
        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )

        return outputs