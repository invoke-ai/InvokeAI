import einops
import torch

from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
from invokeai.backend.flux.extensions.xlabs_ip_adapter_extension import XLabsIPAdapterExtension
from invokeai.backend.flux.math import attention
from invokeai.backend.flux.modules.layers import DoubleStreamBlock, SingleStreamBlock


class CustomDoubleStreamBlockProcessor:
    """A class containing a custom implementation of DoubleStreamBlock.forward() with additional features
    (IP-Adapter, etc.).
    """

    @staticmethod
    def _double_stream_block_forward(
        block: DoubleStreamBlock,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        pe: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """This function is a direct copy of DoubleStreamBlock.forward(), but it returns some of the intermediate
        values.
        """
        img_mod1, img_mod2 = block.img_mod(vec)
        txt_mod1, txt_mod2 = block.txt_mod(vec)

        # prepare image for attention
        img_modulated = block.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = block.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = einops.rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=block.num_heads)
        img_q, img_k = block.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = block.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = block.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = einops.rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=block.num_heads)
        txt_q, txt_k = block.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe, attn_mask=attn_mask)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * block.img_attn.proj(img_attn)
        img = img + img_mod2.gate * block.img_mlp((1 + img_mod2.scale) * block.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * block.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * block.txt_mlp((1 + txt_mod2.scale) * block.txt_norm2(txt) + txt_mod2.shift)
        return img, txt, img_q

    @staticmethod
    def custom_double_block_forward(
        timestep_index: int,
        total_num_timesteps: int,
        block_index: int,
        block: DoubleStreamBlock,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        pe: torch.Tensor,
        ip_adapter_extensions: list[XLabsIPAdapterExtension],
        regional_prompting_extension: RegionalPromptingExtension,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A custom implementation of DoubleStreamBlock.forward() with additional features:
        - IP-Adapter support
        """
        attn_mask = regional_prompting_extension.get_double_stream_attn_mask(block_index)
        img, txt, img_q = CustomDoubleStreamBlockProcessor._double_stream_block_forward(
            block, img, txt, vec, pe, attn_mask=attn_mask
        )

        # Apply IP-Adapter conditioning.
        for ip_adapter_extension in ip_adapter_extensions:
            img = ip_adapter_extension.run_ip_adapter(
                timestep_index=timestep_index,
                total_num_timesteps=total_num_timesteps,
                block_index=block_index,
                block=block,
                img_q=img_q,
                img=img,
            )

        return img, txt


class CustomSingleStreamBlockProcessor:
    """A class containing a custom implementation of SingleStreamBlock.forward() with additional features (masking,
    etc.)
    """

    @staticmethod
    def _single_stream_block_forward(
        block: SingleStreamBlock,
        x: torch.Tensor,
        vec: torch.Tensor,
        pe: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """This function is a direct copy of SingleStreamBlock.forward()."""
        mod, _ = block.modulation(vec)
        x_mod = (1 + mod.scale) * block.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(block.linear1(x_mod), [3 * block.hidden_size, block.mlp_hidden_dim], dim=-1)

        q, k, v = einops.rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=block.num_heads)
        q, k = block.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe, attn_mask=attn_mask)
        # compute activation in mlp stream, cat again and run second linear layer
        output = block.linear2(torch.cat((attn, block.mlp_act(mlp)), 2))
        return x + mod.gate * output

    @staticmethod
    def custom_single_block_forward(
        timestep_index: int,
        total_num_timesteps: int,
        block_index: int,
        block: SingleStreamBlock,
        img: torch.Tensor,
        vec: torch.Tensor,
        pe: torch.Tensor,
        regional_prompting_extension: RegionalPromptingExtension,
    ) -> torch.Tensor:
        """A custom implementation of SingleStreamBlock.forward() with additional features:
        - Masking
        """
        attn_mask = regional_prompting_extension.get_single_stream_attn_mask(block_index)
        return CustomSingleStreamBlockProcessor._single_stream_block_forward(block, img, vec, pe, attn_mask=attn_mask)
