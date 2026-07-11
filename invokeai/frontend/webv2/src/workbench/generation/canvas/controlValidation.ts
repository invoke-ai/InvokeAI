export type ControlAdapterKind = 'controlnet' | 't2i_adapter' | 'control_lora';

export type ControlValidationReason =
  | 'missing_model'
  | 'unsupported_adapter'
  | 'incompatible_base'
  | 'incompatible_adapter'
  | 'control_lora_limit'
  | 'flux_fill_control_lora';

export const isControlKindSupportedForBase = (base: string, kind: ControlAdapterKind): boolean => {
  if (kind === 'controlnet') {
    return base === 'sd-1' || base === 'sdxl' || base === 'flux';
  }
  if (kind === 't2i_adapter') {
    return base === 'sd-1' || base === 'sdxl';
  }
  return base === 'flux';
};

export const getControlValidationReason = (params: {
  adapterModel: { base: string; type: string } | null;
  controlLoraIndex: number;
  kind: ControlAdapterKind;
  mainBase: string;
  mainVariant?: string;
}): ControlValidationReason | null => {
  const { adapterModel, controlLoraIndex, kind, mainBase, mainVariant } = params;
  if (!adapterModel) {
    return 'missing_model';
  }
  if (!isControlKindSupportedForBase(mainBase, kind)) {
    return 'unsupported_adapter';
  }
  if (adapterModel.base !== mainBase) {
    return 'incompatible_base';
  }
  if (adapterModel.type !== kind) {
    return 'incompatible_adapter';
  }
  if (kind === 'control_lora' && controlLoraIndex > 0) {
    return 'control_lora_limit';
  }
  if (kind === 'control_lora' && mainBase === 'flux' && mainVariant === 'dev_fill') {
    return 'flux_fill_control_lora';
  }
  return null;
};
