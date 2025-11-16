import type { Selector } from '@reduxjs/toolkit';
import { EMPTY_ARRAY } from 'app/store/constants';
import type { RootState } from 'app/store/store';
import { useMemo } from 'react';
import {
  modelConfigsAdapterSelectors,
  selectModelConfigsQuery,
  useGetModelConfigsQuery,
} from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';
import {
  isCLIPEmbedModelConfigOrSubmodel,
  isControlLayerModelConfig,
  isControlNetModelConfig,
  isFluxKontextModelConfig,
  isFluxReduxModelConfig,
  isFluxVAEModelConfig,
  isIPAdapterModelConfig,
  isLoRAModelConfig,
  isNonRefinerMainModelConfig,
  isRefinerMainModelModelConfig,
  isSpandrelImageToImageModelConfig,
  isT5EncoderModelConfigOrSubmodel,
  isTIModelConfig,
  isVAEModelConfigOrSubmodel,
} from 'services/api/types';

const buildModelsHook =
  <T extends AnyModelConfig>(typeGuard: (config: AnyModelConfig) => config is T) =>
  (filter: (config: T) => boolean = () => true) => {
    const result = useGetModelConfigsQuery(undefined);
    const modelConfigs = useMemo(() => {
      if (!result.data) {
        return EMPTY_ARRAY;
      }

      return modelConfigsAdapterSelectors
        .selectAll(result.data)
        .filter((config) => typeGuard(config))
        .filter(filter);
    }, [filter, result.data]);

    return [modelConfigs, result] as const;
  };
export const useMainModels = buildModelsHook(isNonRefinerMainModelConfig);
export const useRefinerModels = buildModelsHook(isRefinerMainModelModelConfig);
export const useLoRAModels = buildModelsHook(isLoRAModelConfig);
export const useControlLayerModels = buildModelsHook(isControlLayerModelConfig);
export const useControlNetModels = buildModelsHook(isControlNetModelConfig);
export const useT5EncoderModels = () => buildModelsHook(isT5EncoderModelConfigOrSubmodel)();
export const useCLIPEmbedModels = () => buildModelsHook(isCLIPEmbedModelConfigOrSubmodel)();
export const useSpandrelImageToImageModels = buildModelsHook(isSpandrelImageToImageModelConfig);
export const useEmbeddingModels = buildModelsHook(isTIModelConfig);
export const useVAEModels = () => buildModelsHook(isVAEModelConfigOrSubmodel)();
export const useFluxVAEModels = () => buildModelsHook(isFluxVAEModelConfig)();
export const useGlobalReferenceImageModels = buildModelsHook(
  (config) => isIPAdapterModelConfig(config) || isFluxReduxModelConfig(config) || isFluxKontextModelConfig(config)
);
export const useRegionalReferenceImageModels = buildModelsHook(
  (config) => isIPAdapterModelConfig(config) || isFluxReduxModelConfig(config)
);

const buildModelsSelector =
  <T extends AnyModelConfig>(typeGuard: (config: AnyModelConfig) => config is T): Selector<RootState, T[]> =>
  (state) => {
    const result = selectModelConfigsQuery(state);
    if (!result.data) {
      return EMPTY_ARRAY;
    }
    return modelConfigsAdapterSelectors.selectAll(result.data).filter(typeGuard);
  };
export const selectIPAdapterModels = buildModelsSelector(isIPAdapterModelConfig);
export const selectGlobalRefImageModels = buildModelsSelector(
  (config) => isIPAdapterModelConfig(config) || isFluxReduxModelConfig(config) || isFluxKontextModelConfig(config)
);
export const selectRegionalRefImageModels = buildModelsSelector(
  (config) => isIPAdapterModelConfig(config) || isFluxReduxModelConfig(config)
);
