import { createSelector, type Selector } from '@reduxjs/toolkit';
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
  isCLIPEmbedModelConfig,
  isCLIPVisionModelConfig,
  isControlLayerModelConfig,
  isControlLoRAModelConfig,
  isControlNetModelConfig,
  isFluxMainModelModelConfig,
  isFluxReduxModelConfig,
  isFluxVAEModelConfig,
  isIPAdapterModelConfig,
  isLoRAModelConfig,
  isNonSDXLMainModelConfig,
  isRefinerMainModelModelConfig,
  isSD3MainModelModelConfig,
  isSDXLMainModelModelConfig,
  isSigLipModelConfig,
  isSpandrelImageToImageModelConfig,
  isT2IAdapterModelConfig,
  isT5EncoderModelConfig,
  isTIModelConfig,
  isVAEModelConfig,
} from 'services/api/types';

type ModelHookArgs = { excludeSubmodels?: boolean };

const buildModelsHook =
  <T extends AnyModelConfig>(
    typeGuard: (config: AnyModelConfig, excludeSubmodels?: boolean) => config is T,
    excludeSubmodels?: boolean
  ) =>
  (filter: (config: T) => boolean = () => true) => {
    const result = useGetModelConfigsQuery(undefined);
    const modelConfigs = useMemo(() => {
      if (!result.data) {
        return EMPTY_ARRAY;
      }

      return modelConfigsAdapterSelectors
        .selectAll(result.data)
        .filter((config) => typeGuard(config, excludeSubmodels))
        .filter(filter);
    }, [filter, result.data]);

    return [modelConfigs, result] as const;
  };

export const useNonSDXLMainModels = buildModelsHook(isNonSDXLMainModelConfig);
export const useRefinerModels = buildModelsHook(isRefinerMainModelModelConfig);
export const useFluxModels = buildModelsHook(isFluxMainModelModelConfig);
export const useSD3Models = buildModelsHook(isSD3MainModelModelConfig);
export const useSDXLModels = buildModelsHook(isSDXLMainModelModelConfig);
export const useLoRAModels = buildModelsHook(isLoRAModelConfig);
export const useControlLoRAModel = buildModelsHook(isControlLoRAModelConfig);
export const useControlLayerModels = buildModelsHook(isControlLayerModelConfig);
export const useControlNetModels = buildModelsHook(isControlNetModelConfig);
export const useT2IAdapterModels = buildModelsHook(isT2IAdapterModelConfig);
export const useT5EncoderModels = (args?: ModelHookArgs) =>
  buildModelsHook(isT5EncoderModelConfig, args?.excludeSubmodels)();
export const useCLIPEmbedModels = (args?: ModelHookArgs) =>
  buildModelsHook(isCLIPEmbedModelConfig, args?.excludeSubmodels)();
export const useSpandrelImageToImageModels = buildModelsHook(isSpandrelImageToImageModelConfig);
export const useIPAdapterModels = buildModelsHook(isIPAdapterModelConfig);
export const useEmbeddingModels = buildModelsHook(isTIModelConfig);
export const useVAEModels = (args?: ModelHookArgs) => buildModelsHook(isVAEModelConfig, args?.excludeSubmodels)();
export const useFluxVAEModels = (args?: ModelHookArgs) =>
  buildModelsHook(isFluxVAEModelConfig, args?.excludeSubmodels)();
export const useCLIPVisionModels = buildModelsHook(isCLIPVisionModelConfig);
export const useSigLipModels = buildModelsHook(isSigLipModelConfig);
export const useFluxReduxModels = buildModelsHook(isFluxReduxModelConfig);
export const useIPAdapterOrFLUXReduxModels = buildModelsHook(
  (config) => isIPAdapterModelConfig(config) || isFluxReduxModelConfig(config)
);

// const buildModelsSelector =
//   <T extends AnyModelConfig>(typeGuard: (config: AnyModelConfig) => config is T): Selector<RootState, T[]> =>
//   (state) => {
//     const result = selectModelConfigsQuery(state);
//     if (!result.data) {
//       return EMPTY_ARRAY;
//     }
//     return modelConfigsAdapterSelectors.selectAll(result.data).filter(typeGuard);
//   };
// export const selectSDMainModels = buildModelsSelector(isNonRefinerNonFluxMainModelConfig);
// export const selectMainModels = buildModelsSelector(isNonRefinerMainModelConfig);
// export const selectNonSDXLMainModels = buildModelsSelector(isNonSDXLMainModelConfig);
// export const selectRefinerModels = buildModelsSelector(isRefinerMainModelModelConfig);
// export const selectFluxModels = buildModelsSelector(isFluxMainModelModelConfig);
// export const selectSDXLModels = buildModelsSelector(isSDXLMainModelModelConfig);
// export const selectLoRAModels = buildModelsSelector(isLoRAModelConfig);
// export const selectControlNetAndT2IAdapterModels = buildModelsSelector(isControlNetOrT2IAdapterModelConfig);
// export const selectControlNetModels = buildModelsSelector(isControlNetModelConfig);
// export const selectT2IAdapterModels = buildModelsSelector(isT2IAdapterModelConfig);
// export const selectT5EncoderModels = buildModelsSelector(isT5EncoderModelConfig);
// export const selectClipEmbedModels = buildModelsSelector(isClipEmbedModelConfig);
// export const selectSpandrelImageToImageModels = buildModelsSelector(isSpandrelImageToImageModelConfig);
// export const selectIPAdapterModels = buildModelsSelector(isIPAdapterModelConfig);
// export const selectEmbeddingModels = buildModelsSelector(isTIModelConfig);
// export const selectVAEModels = buildModelsSelector(isVAEModelConfig);
// export const selectFluxVAEModels = buildModelsSelector(isFluxVAEModelConfig);

export const buildSelectModelConfig = <T extends AnyModelConfig>(
  key: string,
  typeGuard: (config: AnyModelConfig) => config is T
): Selector<RootState, T | null> =>
  createSelector(selectModelConfigsQuery, (result) => {
    if (!result.data) {
      return null;
    }
    return (
      modelConfigsAdapterSelectors
        .selectAll(result.data)
        .filter(typeGuard)
        .find((m) => m.key === key) ?? null
    );
  });
