import type { Selector } from '@reduxjs/toolkit';
import { EMPTY_ARRAY } from 'app/store/constants';
import type { RootState } from 'app/store/store';
import { useMemo } from 'react';
import {
  modelConfigsAdapterSelectors,
  selectMissingModelsQuery,
  selectModelConfigsQuery,
  useGetMissingModelsQuery,
  useGetModelConfigsQuery,
} from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';
import {
  isCLIPEmbedModelConfigOrSubmodel,
  isControlLayerModelConfig,
  isControlNetModelConfig,
  isFlux1VAEModelConfig,
  isFlux2VAEModelConfig,
  isFluxKontextModelConfig,
  isFluxReduxModelConfig,
  isFluxVAEModelConfig,
  isIPAdapterModelConfig,
  isLoRAModelConfig,
  isNonRefinerMainModelConfig,
  isQwen3EncoderModelConfig,
  isRefinerMainModelModelConfig,
  isSpandrelImageToImageModelConfig,
  isT5EncoderModelConfigOrSubmodel,
  isTIModelConfig,
  isVAEModelConfigOrSubmodel,
  isZImageDiffusersMainModelConfig,
} from 'services/api/types';

const buildModelsHook =
  <T extends AnyModelConfig>(typeGuard: (config: AnyModelConfig) => config is T) =>
  (filter: (config: T) => boolean = () => true) => {
    const result = useGetModelConfigsQuery(undefined);
    const { data: missingModelsData } = useGetMissingModelsQuery();

    const modelConfigs = useMemo(() => {
      if (!result.data) {
        return EMPTY_ARRAY;
      }

      // Get set of missing model keys to exclude from selection
      const missingModelKeys = new Set(
        modelConfigsAdapterSelectors.selectAll(missingModelsData ?? { ids: [], entities: {} }).map((m) => m.key)
      );

      return modelConfigsAdapterSelectors
        .selectAll(result.data)
        .filter((config) => typeGuard(config))
        .filter((config) => !missingModelKeys.has(config.key))
        .filter(filter);
    }, [filter, result.data, missingModelsData]);

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
export const useFlux1VAEModels = () => buildModelsHook(isFlux1VAEModelConfig)();
export const useFlux2VAEModels = () => buildModelsHook(isFlux2VAEModelConfig)();
export const useZImageDiffusersModels = () => buildModelsHook(isZImageDiffusersMainModelConfig)();
export const useQwen3EncoderModels = () => buildModelsHook(isQwen3EncoderModelConfig)();
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

    // Get set of missing model keys to exclude from selection
    const missingResult = selectMissingModelsQuery(state);
    const missingModelKeys = new Set(
      modelConfigsAdapterSelectors.selectAll(missingResult.data ?? { ids: [], entities: {} }).map((m) => m.key)
    );

    return modelConfigsAdapterSelectors
      .selectAll(result.data)
      .filter(typeGuard)
      .filter((config) => !missingModelKeys.has(config.key));
  };
export const selectIPAdapterModels = buildModelsSelector(isIPAdapterModelConfig);
export const selectGlobalRefImageModels = buildModelsSelector(
  (config) => isIPAdapterModelConfig(config) || isFluxReduxModelConfig(config) || isFluxKontextModelConfig(config)
);
export const selectRegionalRefImageModels = buildModelsSelector(
  (config) => isIPAdapterModelConfig(config) || isFluxReduxModelConfig(config)
);
export const selectQwen3EncoderModels = buildModelsSelector(isQwen3EncoderModelConfig);
export const selectZImageDiffusersModels = buildModelsSelector(isZImageDiffusersMainModelConfig);
export const selectFluxVAEModels = buildModelsSelector(isFluxVAEModelConfig);
