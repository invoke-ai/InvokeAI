import { getStore } from 'app/store/nanostores/store';
import type { ControlNetConfig, IPAdapterConfig, T2IAdapterConfig } from 'features/controlAdapters/store/types';
import type { LoRA } from 'features/lora/store/loraSlice';
import type { MetadataValidateFunc } from 'features/metadata/types';
import { InvalidModelConfigError } from 'features/parameters/util/modelFetchingHelpers';
import type { BaseModelType, RefinerMainModelConfig, VAEModelConfig } from 'services/api/types';

/**
 * Checks the given base model type against the currently-selected model's base type and throws an error if they are
 * incompatible.
 * @param base The base model type to validate.
 * @param message An optional message to use in the error if the base model is incompatible.
 */
const validateBaseCompatibility = (base?: BaseModelType, message?: string) => {
  if (!base) {
    throw new InvalidModelConfigError(message || 'Missing base');
  }
  const currentBase = getStore().getState().generation.model?.base;
  if (currentBase && base !== currentBase) {
    throw new InvalidModelConfigError(message || `Incompatible base models: ${base} and ${currentBase}`);
  }
};

const validateRefinerModel: MetadataValidateFunc<RefinerMainModelConfig> = (refinerModel) => {
  validateBaseCompatibility('sdxl', 'Refiner incompatible with currently-selected model');
  return new Promise((resolve) => resolve(refinerModel));
};

const validateVAEModel: MetadataValidateFunc<VAEModelConfig> = (vaeModel) => {
  validateBaseCompatibility(vaeModel.base, 'VAE incompatible with currently-selected model');
  return new Promise((resolve) => resolve(vaeModel));
};

const validateLoRA: MetadataValidateFunc<LoRA> = (lora) => {
  validateBaseCompatibility(lora.model.base, 'LoRA incompatible with currently-selected model');
  return new Promise((resolve) => resolve(lora));
};

const validateLoRAs: MetadataValidateFunc<LoRA[]> = (loras) => {
  const validatedLoRAs: LoRA[] = [];
  loras.forEach((lora) => {
    try {
      validateBaseCompatibility(lora.model.base, 'LoRA incompatible with currently-selected model');
      validatedLoRAs.push(lora);
    } catch {
      // This is a no-op - we want to continue validating the rest of the LoRAs, and an empty list is valid.
    }
  });
  return new Promise((resolve) => resolve(validatedLoRAs));
};

const validateControlNet: MetadataValidateFunc<ControlNetConfig> = (controlNet) => {
  validateBaseCompatibility(controlNet.model?.base, 'ControlNet incompatible with currently-selected model');
  return new Promise((resolve) => resolve(controlNet));
};

const validateControlNets: MetadataValidateFunc<ControlNetConfig[]> = (controlNets) => {
  const validatedControlNets: ControlNetConfig[] = [];
  controlNets.forEach((controlNet) => {
    try {
      validateBaseCompatibility(controlNet.model?.base, 'ControlNet incompatible with currently-selected model');
      validatedControlNets.push(controlNet);
    } catch {
      // This is a no-op - we want to continue validating the rest of the ControlNets, and an empty list is valid.
    }
  });
  return new Promise((resolve) => resolve(validatedControlNets));
};

const validateT2IAdapter: MetadataValidateFunc<T2IAdapterConfig> = (t2iAdapter) => {
  validateBaseCompatibility(t2iAdapter.model?.base, 'T2I Adapter incompatible with currently-selected model');
  return new Promise((resolve) => resolve(t2iAdapter));
};

const validateT2IAdapters: MetadataValidateFunc<T2IAdapterConfig[]> = (t2iAdapters) => {
  const validatedT2IAdapters: T2IAdapterConfig[] = [];
  t2iAdapters.forEach((t2iAdapter) => {
    try {
      validateBaseCompatibility(t2iAdapter.model?.base, 'T2I Adapter incompatible with currently-selected model');
      validatedT2IAdapters.push(t2iAdapter);
    } catch {
      // This is a no-op - we want to continue validating the rest of the T2I Adapters, and an empty list is valid.
    }
  });
  return new Promise((resolve) => resolve(validatedT2IAdapters));
};

const validateIPAdapter: MetadataValidateFunc<IPAdapterConfig> = (ipAdapter) => {
  validateBaseCompatibility(ipAdapter.model?.base, 'IP Adapter incompatible with currently-selected model');
  return new Promise((resolve) => resolve(ipAdapter));
};

const validateIPAdapters: MetadataValidateFunc<IPAdapterConfig[]> = (ipAdapters) => {
  const validatedIPAdapters: IPAdapterConfig[] = [];
  ipAdapters.forEach((ipAdapter) => {
    try {
      validateBaseCompatibility(ipAdapter.model?.base, 'IP Adapter incompatible with currently-selected model');
      validatedIPAdapters.push(ipAdapter);
    } catch {
      // This is a no-op - we want to continue validating the rest of the IP Adapters, and an empty list is valid.
    }
  });
  return new Promise((resolve) => resolve(validatedIPAdapters));
};

export const validators = {
  refinerModel: validateRefinerModel,
  vaeModel: validateVAEModel,
  lora: validateLoRA,
  loras: validateLoRAs,
  controlNet: validateControlNet,
  controlNets: validateControlNets,
  t2iAdapter: validateT2IAdapter,
  t2iAdapters: validateT2IAdapters,
  ipAdapter: validateIPAdapter,
  ipAdapters: validateIPAdapters,
} as const;
