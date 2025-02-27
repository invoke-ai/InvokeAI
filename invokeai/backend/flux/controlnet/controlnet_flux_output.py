from dataclasses import dataclass

import torch


@dataclass
class ControlNetFluxOutput:
    single_block_residuals: list[torch.Tensor] | None
    double_block_residuals: list[torch.Tensor] | None

    def apply_weight(self, weight: float):
        if self.single_block_residuals is not None:
            for i in range(len(self.single_block_residuals)):
                self.single_block_residuals[i] = self.single_block_residuals[i] * weight
        if self.double_block_residuals is not None:
            for i in range(len(self.double_block_residuals)):
                self.double_block_residuals[i] = self.double_block_residuals[i] * weight


def add_tensor_lists_elementwise(
    list1: list[torch.Tensor] | None, list2: list[torch.Tensor] | None
) -> list[torch.Tensor] | None:
    """Add two tensor lists elementwise that could be None."""
    if list1 is None and list2 is None:
        return None
    if list1 is None:
        return list2
    if list2 is None:
        return list1

    new_list: list[torch.Tensor] = []
    for list1_tensor, list2_tensor in zip(list1, list2, strict=True):
        new_list.append(list1_tensor + list2_tensor)
    return new_list


def add_controlnet_flux_outputs(
    controlnet_output_1: ControlNetFluxOutput, controlnet_output_2: ControlNetFluxOutput
) -> ControlNetFluxOutput:
    return ControlNetFluxOutput(
        single_block_residuals=add_tensor_lists_elementwise(
            controlnet_output_1.single_block_residuals, controlnet_output_2.single_block_residuals
        ),
        double_block_residuals=add_tensor_lists_elementwise(
            controlnet_output_1.double_block_residuals, controlnet_output_2.double_block_residuals
        ),
    )


def sum_controlnet_flux_outputs(
    controlnet_outputs: list[ControlNetFluxOutput],
) -> ControlNetFluxOutput:
    controlnet_output_sum = ControlNetFluxOutput(single_block_residuals=None, double_block_residuals=None)

    for controlnet_output in controlnet_outputs:
        controlnet_output_sum = add_controlnet_flux_outputs(controlnet_output_sum, controlnet_output)

    return controlnet_output_sum
