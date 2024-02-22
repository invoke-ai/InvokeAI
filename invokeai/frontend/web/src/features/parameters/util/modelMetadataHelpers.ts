import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import type { ControlNetConfig, IPAdapterConfig, T2IAdapterConfig } from 'features/controlAdapters/store/types';
import {
  initialControlNet,
  initialIPAdapter,
  initialT2IAdapter,
} from 'features/controlAdapters/util/buildControlAdapter';
import type { LoRA } from 'features/lora/store/loraSlice';
import type { ModelIdentifierWithBase } from 'features/nodes/types/common';
import { zModelIdentifierWithBase } from 'features/nodes/types/common';
import type {
  ControlNetMetadataItem,
  IPAdapterMetadataItem,
  LoRAMetadataItem,
  T2IAdapterMetadataItem,
} from 'features/nodes/types/metadata';
import {
  fetchControlNetModel,
  fetchIPAdapterModel,
  fetchLoRAModel,
  fetchMainModel,
  fetchRefinerModel,
  fetchT2IAdapterModel,
  fetchVAEModel,
  getModelKey,
  raiseIfBaseIncompatible,
} from 'features/parameters/util/modelFetchingHelpers';
import type { BaseModelType } from 'services/api/types';
import { v4 as uuidv4 } from 'uuid';

export const prepareMainModelMetadataItem = async (model: unknown): Promise<ModelIdentifierWithBase> => {
  const key = getModelKey(model);
  const mainModel = await fetchMainModel(key);
  return zModelIdentifierWithBase.parse(mainModel);
};

export const prepareRefinerMetadataItem = async (model: unknown): Promise<ModelIdentifierWithBase> => {
  const key = getModelKey(model);
  const refinerModel = await fetchRefinerModel(key);
  return zModelIdentifierWithBase.parse(refinerModel);
};

export const prepareVAEMetadataItem = async (vae: unknown, base?: BaseModelType): Promise<ModelIdentifierWithBase> => {
  const key = getModelKey(vae);
  const vaeModel = await fetchVAEModel(key);
  raiseIfBaseIncompatible(vaeModel.base, base, 'VAE incompatible with currently-selected model');
  return zModelIdentifierWithBase.parse(vaeModel);
};

export const prepareLoRAMetadataItem = async (
  loraMetadataItem: LoRAMetadataItem,
  base?: BaseModelType
): Promise<LoRA> => {
  const key = getModelKey(loraMetadataItem.lora);
  const loraModel = await fetchLoRAModel(key);
  raiseIfBaseIncompatible(loraModel.base, base, 'LoRA incompatible with currently-selected model');
  return { key: loraModel.key, base: loraModel.base, weight: loraMetadataItem.weight, isEnabled: true };
};

export const prepareControlNetMetadataItem = async (
  controlnetMetadataItem: ControlNetMetadataItem,
  base?: BaseModelType
): Promise<ControlNetConfig> => {
  const key = getModelKey(controlnetMetadataItem.control_model);
  const controlNetModel = await fetchControlNetModel(key);
  raiseIfBaseIncompatible(controlNetModel.base, base, 'ControlNet incompatible with currently-selected model');

  const { image, control_weight, begin_step_percent, end_step_percent, control_mode, resize_mode } =
    controlnetMetadataItem;

  // We don't save the original image that was processed into a control image, only the processed image
  const processorType = 'none';
  const processorNode = CONTROLNET_PROCESSORS.none.default;

  const controlnet: ControlNetConfig = {
    type: 'controlnet',
    isEnabled: true,
    model: zModelIdentifierWithBase.parse(controlNetModel),
    weight: typeof control_weight === 'number' ? control_weight : initialControlNet.weight,
    beginStepPct: begin_step_percent || initialControlNet.beginStepPct,
    endStepPct: end_step_percent || initialControlNet.endStepPct,
    controlMode: control_mode || initialControlNet.controlMode,
    resizeMode: resize_mode || initialControlNet.resizeMode,
    controlImage: image?.image_name || null,
    processedControlImage: image?.image_name || null,
    processorType,
    processorNode,
    shouldAutoConfig: true,
    id: uuidv4(),
  };

  return controlnet;
};

export const prepareT2IAdapterMetadataItem = async (
  t2iAdapterMetadataItem: T2IAdapterMetadataItem,
  base?: BaseModelType
): Promise<T2IAdapterConfig> => {
  const key = getModelKey(t2iAdapterMetadataItem.t2i_adapter_model);
  const t2iAdapterModel = await fetchT2IAdapterModel(key);
  raiseIfBaseIncompatible(t2iAdapterModel.base, base, 'T2I Adapter incompatible with currently-selected model');

  const { image, weight, begin_step_percent, end_step_percent, resize_mode } = t2iAdapterMetadataItem;

  // We don't save the original image that was processed into a control image, only the processed image
  const processorType = 'none';
  const processorNode = CONTROLNET_PROCESSORS.none.default;

  const t2iAdapter: T2IAdapterConfig = {
    type: 't2i_adapter',
    isEnabled: true,
    model: zModelIdentifierWithBase.parse(t2iAdapterModel),
    weight: typeof weight === 'number' ? weight : initialT2IAdapter.weight,
    beginStepPct: begin_step_percent || initialT2IAdapter.beginStepPct,
    endStepPct: end_step_percent || initialT2IAdapter.endStepPct,
    resizeMode: resize_mode || initialT2IAdapter.resizeMode,
    controlImage: image?.image_name || null,
    processedControlImage: image?.image_name || null,
    processorType,
    processorNode,
    shouldAutoConfig: true,
    id: uuidv4(),
  };

  return t2iAdapter;
};

export const prepareIPAdapterMetadataItem = async (
  ipAdapterMetadataItem: IPAdapterMetadataItem,
  base?: BaseModelType
): Promise<IPAdapterConfig> => {
  const key = getModelKey(ipAdapterMetadataItem?.ip_adapter_model);
  const ipAdapterModel = await fetchIPAdapterModel(key);
  raiseIfBaseIncompatible(ipAdapterModel.base, base, 'T2I Adapter incompatible with currently-selected model');

  const { image, weight, begin_step_percent, end_step_percent } = ipAdapterMetadataItem;

  const ipAdapter: IPAdapterConfig = {
    id: uuidv4(),
    type: 'ip_adapter',
    isEnabled: true,
    controlImage: image?.image_name ?? null,
    model: zModelIdentifierWithBase.parse(ipAdapterModel),
    weight: weight ?? initialIPAdapter.weight,
    beginStepPct: begin_step_percent ?? initialIPAdapter.beginStepPct,
    endStepPct: end_step_percent ?? initialIPAdapter.endStepPct,
  };

  return ipAdapter;
};
