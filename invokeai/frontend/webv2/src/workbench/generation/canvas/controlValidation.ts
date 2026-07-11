export type ControlAdapterKind = 'controlnet' | 't2i_adapter' | 'control_lora' | 'z_image_control';

export type ControlValidationReason =
  | 'missing_model'
  | 'unsupported_adapter'
  | 'incompatible_base'
  | 'incompatible_adapter'
  | 'invalid_adapter_values'
  | 'control_lora_limit'
  | 'z_image_control_limit'
  | 'flux_fill_control_lora';

export const isControlKindSupportedForBase = (base: string, kind: ControlAdapterKind): boolean => {
  if (kind === 'controlnet') {
    return base === 'sd-1' || base === 'sdxl' || base === 'flux';
  }
  if (kind === 't2i_adapter') {
    return base === 'sd-1' || base === 'sdxl';
  }
  if (kind === 'z_image_control') {
    return base === 'z-image';
  }
  return base === 'flux';
};

export const getControlValidationReason = (params: {
  adapterModel: { base: string; type: string } | null;
  beginEndStepPct: [number, number];
  controlLoraIndex: number;
  kind: ControlAdapterKind;
  mainBase: string;
  mainVariant?: string;
  weight: number;
  zImageControlIndex?: number;
}): ControlValidationReason | null => {
  const {
    adapterModel,
    beginEndStepPct,
    controlLoraIndex,
    kind,
    mainBase,
    mainVariant,
    weight,
    zImageControlIndex = 0,
  } = params;
  if (!areControlAdapterValuesValid(kind, weight, beginEndStepPct)) {
    return 'invalid_adapter_values';
  }
  if (!adapterModel) {
    return 'missing_model';
  }
  if (!isControlKindSupportedForBase(mainBase, kind)) {
    return 'unsupported_adapter';
  }
  if (adapterModel.base !== mainBase) {
    return 'incompatible_base';
  }
  const expectedModelType = kind === 'z_image_control' ? 'controlnet' : kind;
  if (adapterModel.type !== expectedModelType) {
    return 'incompatible_adapter';
  }
  if (kind === 'control_lora' && controlLoraIndex > 0) {
    return 'control_lora_limit';
  }
  if (kind === 'z_image_control' && zImageControlIndex > 0) {
    return 'z_image_control_limit';
  }
  if (kind === 'control_lora' && mainBase === 'flux' && mainVariant === 'dev_fill') {
    return 'flux_fill_control_lora';
  }
  return null;
};
import { areControlAdapterValuesValid } from '@workbench/controlAdapters';
