import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Input, InputField, InvocationContext, invocation
from invokeai.app.invocations.compel import ConditioningFieldData
from invokeai.app.invocations.primitives import ConditioningField, ConditioningOutput
from invokeai.app.shared.fields import FieldDescriptions
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import BasicConditioningInfo, ExtraConditioningInfo


@invocation(
    "conditioning_concat",
    title="Conditioning Concat",
    tags=["conditioning", "concat"],
    category="conditioning",
    version="1.0.0",
)
class ConditioningConcatInvocation(BaseInvocation):
    """Concat two different conditionings and output the result"""

    cond1: ConditioningField = InputField(
        description=FieldDescriptions.cond, input=Input.Connection, title="Conditioning 1", ui_order=0
    )
    cond2: ConditioningField = InputField(
        description=FieldDescriptions.cond, input=Input.Connection, title="Conditioning 2", ui_order=1
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ConditioningOutput:
        cond_1 = context.services.latents.get(self.cond1.conditioning_name)
        c1 = cond_1.conditionings[0].to(device="cpu", dtype=torch.float16)
        c1_extra_conditioning_info = c1.extra_conditioning

        cond_2 = context.services.latents.get(self.cond2.conditioning_name)
        c2 = cond_2.conditionings[0].to(device="cpu", dtype=torch.float16)
        c2_extra_conditioning_info = c2.extra_conditioning

        c_concat = torch.cat((c1.embeds, c2.embeds), dim=1)

        ec_concat_tokens_count_including_eos_bos = (
            c1_extra_conditioning_info.tokens_count_including_eos_bos
            + c2_extra_conditioning_info.tokens_count_including_eos_bos
        )
        ec_cross_attention_control_args = None

        ec_concat = ExtraConditioningInfo(
            tokens_count_including_eos_bos=ec_concat_tokens_count_including_eos_bos,
            cross_attention_control_args=ec_cross_attention_control_args,
        )

        conditioning_data = ConditioningFieldData(
            conditionings=[
                BasicConditioningInfo(
                    embeds=c_concat,
                    extra_conditioning=ec_concat,
                )
            ]
        )

        print(conditioning_data)

        conditioning_name = f"{context.graph_execution_state_id}_{self.id}_conditioning"
        context.services.latents.save(conditioning_name, conditioning_data)

        return ConditioningOutput(
            conditioning=ConditioningField(
                conditioning_name=conditioning_name,
            ),
        )
