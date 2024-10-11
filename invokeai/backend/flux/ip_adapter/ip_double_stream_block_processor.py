# This file is based on:
# https://github.com/XLabs-AI/x-flux/blob/47495425dbed499be1e8e5a6e52628b07349cba2/src/flux/modules/layers.py#L221


class IPDoubleStreamBlockProcessor(nn.Module):
    """Attention processor for handling IP-adapter with double stream block."""

    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("IPDoubleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch.")

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter
        self.ip_adapter_double_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=True)

        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.bias)

        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.bias)

    def __call__(self, attn, img, txt, vec, pe, image_proj, ip_scale=1.0, **attention_kwargs):
        # Prepare image for attention
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # print(f"txt_attn shape: {txt_attn.size()}")
        # print(f"img_attn shape: {img_attn.size()}")

        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        # IP-adapter processing
        ip_query = img_q  # latent sample query
        ip_key = self.ip_adapter_double_stream_k_proj(image_proj)
        ip_value = self.ip_adapter_double_stream_v_proj(image_proj)

        # Reshape projections for multi-head attention
        ip_key = rearrange(ip_key, "B L (H D) -> B H L D", H=attn.num_heads, D=attn.head_dim)
        ip_value = rearrange(ip_value, "B L (H D) -> B H L D", H=attn.num_heads, D=attn.head_dim)

        # Compute attention between IP projections and the latent query
        ip_attention = F.scaled_dot_product_attention(ip_query, ip_key, ip_value, dropout_p=0.0, is_causal=False)
        ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads, D=attn.head_dim)

        img = img + ip_scale * ip_attention

        return img, txt
