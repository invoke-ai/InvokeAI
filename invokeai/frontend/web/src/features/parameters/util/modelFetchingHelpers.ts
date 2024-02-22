import { getStore } from 'app/store/nanostores/store';
import { isModelIdentifier } from 'features/nodes/types/common';
import { modelsApi } from 'services/api/endpoints/models';
import type { AnyModelConfig, BaseModelType } from 'services/api/types';
import {
  isControlNetModelConfig,
  isIPAdapterModelConfig,
  isLoRAModelConfig,
  isNonRefinerMainModelConfig,
  isRefinerMainModelModelConfig,
  isT2IAdapterModelConfig,
  isTextualInversionModelConfig,
  isVAEModelConfig,
} from 'services/api/types';

/**
 * Raised when a model config is unable to be fetched.
 */
export class ModelConfigNotFoundError extends Error {
  /**
   * Create ModelConfigNotFoundError
   * @param {String} message
   */
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
  }
}

/**
 * Raised when a fetched model config is of an unexpected type.
 */
export class InvalidModelConfigError extends Error {
  /**
   * Create InvalidModelConfigError
   * @param {String} message
   */
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
  }
}

export const fetchModelConfig = async (key: string): Promise<AnyModelConfig> => {
  const { dispatch } = getStore();
  try {
    const req = dispatch(modelsApi.endpoints.getModelConfig.initiate(key));
    req.unsubscribe();
    return await req.unwrap();
  } catch {
    throw new ModelConfigNotFoundError(`Unable to retrieve model config for key ${key}`);
  }
};

export const fetchModelConfigWithTypeGuard = async <T extends AnyModelConfig>(
  key: string,
  typeGuard: (config: AnyModelConfig) => config is T
) => {
  const modelConfig = await fetchModelConfig(key);
  if (!typeGuard(modelConfig)) {
    throw new InvalidModelConfigError(`Invalid model type for key ${key}: ${modelConfig.type}`);
  }
  return modelConfig;
};

export const fetchMainModel = async (key: string) => {
  return fetchModelConfigWithTypeGuard(key, isNonRefinerMainModelConfig);
};

export const fetchRefinerModel = async (key: string) => {
  return fetchModelConfigWithTypeGuard(key, isRefinerMainModelModelConfig);
};

export const fetchVAEModel = async (key: string) => {
  return fetchModelConfigWithTypeGuard(key, isVAEModelConfig);
};

export const fetchLoRAModel = async (key: string) => {
  return fetchModelConfigWithTypeGuard(key, isLoRAModelConfig);
};

export const fetchControlNetModel = async (key: string) => {
  return fetchModelConfigWithTypeGuard(key, isControlNetModelConfig);
};

export const fetchIPAdapterModel = async (key: string) => {
  return fetchModelConfigWithTypeGuard(key, isIPAdapterModelConfig);
};

export const fetchT2IAdapterModel = async (key: string) => {
  return fetchModelConfigWithTypeGuard(key, isT2IAdapterModelConfig);
};

export const fetchTextualInversionModel = async (key: string) => {
  return fetchModelConfigWithTypeGuard(key, isTextualInversionModelConfig);
};

export const isBaseCompatible = (sourceBase: BaseModelType, targetBase: BaseModelType) => {
  return sourceBase === targetBase;
};

export const raiseIfBaseIncompatible = (sourceBase: BaseModelType, targetBase?: BaseModelType, message?: string) => {
  if (targetBase && !isBaseCompatible(sourceBase, targetBase)) {
    throw new InvalidModelConfigError(message || `Incompatible base models: ${sourceBase} and ${targetBase}`);
  }
};

export const getModelKey = (modelIdentifier: unknown, message?: string): string => {
  if (!isModelIdentifier(modelIdentifier)) {
    throw new InvalidModelConfigError(message || `Invalid model identifier: ${modelIdentifier}`);
  }
  return modelIdentifier.key;
};
