import { getStore } from 'app/store/nanostores/store';
import type { ModelIdentifierField } from 'features/nodes/types/common';
import { isModelIdentifier, isModelIdentifierV2 } from 'features/nodes/types/common';
import type { ModelIdentifier } from 'features/nodes/types/v2/common';
import { modelsApi } from 'services/api/endpoints/models';
import type { AnyModelConfig, BaseModelType, ModelType } from 'services/api/types';

/**
 * Raised when a model config is unable to be fetched.
 */
class ModelConfigNotFoundError extends Error {
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

/**
 * Fetches the model config for a given model key.
 * @param key The model key.
 * @returns A promise that resolves to the model config.
 * @throws {ModelConfigNotFoundError} If the model config is unable to be fetched.
 */
export const fetchModelConfig = async (key: string): Promise<AnyModelConfig> => {
  const { dispatch } = getStore();
  try {
    const req = dispatch(modelsApi.endpoints.getModelConfig.initiate(key, { subscribe: false }));
    return await req.unwrap();
  } catch {
    throw new ModelConfigNotFoundError(`Unable to retrieve model config for key ${key}`);
  }
};

/**
 * Fetches the model config for a given model name, base model, and model type. This provides backwards compatibility
 * for MM1 model identifiers.
 * @param name The model name.
 * @param base The base model.
 * @param type The model type.
 * @returns A promise that resolves to the model config.
 * @throws {ModelConfigNotFoundError} If the model config is unable to be fetched.
 */
const fetchModelConfigByAttrs = async (name: string, base: BaseModelType, type: ModelType): Promise<AnyModelConfig> => {
  const { dispatch } = getStore();
  try {
    const req = dispatch(
      modelsApi.endpoints.getModelConfigByAttrs.initiate({ name, base, type }, { subscribe: false })
    );
    return await req.unwrap();
  } catch {
    throw new ModelConfigNotFoundError(`Unable to retrieve model config for name/base/type ${name}/${base}/${type}`);
  }
};

/**
 * Fetches the model config given an identifier. First attempts to fetch by key, then falls back to fetching by attrs.
 * @param identifier The model identifier.
 * @returns A promise that resolves to the model config.
 * @throws {ModelConfigNotFoundError} If the model config is unable to be fetched.
 */
export const fetchModelConfigByIdentifier = async (identifier: ModelIdentifierField): Promise<AnyModelConfig> => {
  try {
    return await fetchModelConfig(identifier.key);
  } catch {
    try {
      return await fetchModelConfigByAttrs(identifier.name, identifier.base, identifier.type);
    } catch {
      throw new ModelConfigNotFoundError(`Unable to retrieve model config for identifier ${identifier}`);
    }
  }
};

/**
 * Fetches the model config for a given model key and type, and ensures that the model config is of a specific type.
 * @param key The model key.
 * @param typeGuard A type guard function that checks if the model config is of the expected type.
 * @returns A promise that resolves to the model config. The model config is guaranteed to be of the expected type.
 * @throws {InvalidModelConfigError} If the model config is unable to be fetched or is of an unexpected type.
 */
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

/**
 * Fetches the model key from a model identifier. This includes fetching the key for MM1 format model identifiers.
 * @param modelIdentifier The model identifier. This can be a MM1 or MM2 identifier. In every case, we attempt to fetch
 * the model config from the server to ensure that the model identifier is valid and represents an installed model.
 * @param type The type of model to fetch. This is used to fetch the key for MM1 format model identifiers.
 * @param message An optional custom message to include in the error if the model identifier is invalid.
 * @returns A promise that resolves to the model key.
 * @throws {InvalidModelConfigError} If the model identifier is invalid.
 */
export const getModelKey = async (
  modelIdentifier: unknown | ModelIdentifierField | ModelIdentifier,
  type: ModelType,
  message?: string
): Promise<string> => {
  if (isModelIdentifier(modelIdentifier)) {
    try {
      // Check if the model exists by key
      return (await fetchModelConfig(modelIdentifier.key)).key;
    } catch {
      // If not, fetch the model key by name and base model
      return (await fetchModelConfigByAttrs(modelIdentifier.name, modelIdentifier.base, type)).key;
    }
  } else if (isModelIdentifierV2(modelIdentifier)) {
    // Try by old-format model identifier
    return (await fetchModelConfigByAttrs(modelIdentifier.model_name, modelIdentifier.base_model, type)).key;
  }
  // Nope, couldn't find it
  throw new InvalidModelConfigError(message || `Invalid model identifier: ${modelIdentifier}`);
};
