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
  isControlNetModelConfig,
  isControlNetOrT2IAdapterModelConfig,
  isFluxMainModelModelConfig,
  isFluxVAEModelConfig,
  isIPAdapterModelConfig,
  isLoRAModelConfig,
  isNonRefinerMainModelConfig,
  isNonSDXLMainModelConfig,
  isRefinerMainModelModelConfig,
  isSDXLMainModelModelConfig,
  isSpandrelImageToImageModelConfig,
  isT2IAdapterModelConfig,
  isT5EncoderModelConfig,
  isTIModelConfig,
  isVAEModelConfig,
} from 'services/api/types';

const buildModelsHook =
  <T extends AnyModelConfig>(typeGuard: (config: AnyModelConfig) => config is T) =>
  () => {
    const result = useGetModelConfigsQuery(undefined);
    const modelConfigs = useMemo(() => {
      if (!result.data) {
        return EMPTY_ARRAY;
      }

      return modelConfigsAdapterSelectors.selectAll(result.data).filter(typeGuard);
    }, [result]);

    return [modelConfigs, result] as const;
  };

export const useMainModels = buildModelsHook(isNonRefinerMainModelConfig);
export const useNonSDXLMainModels = buildModelsHook(isNonSDXLMainModelConfig);
export const useRefinerModels = buildModelsHook(isRefinerMainModelModelConfig);
export const useFluxModels = buildModelsHook(isFluxMainModelModelConfig);
export const useSDXLModels = buildModelsHook(isSDXLMainModelModelConfig);
export const useLoRAModels = buildModelsHook(isLoRAModelConfig);
export const useControlNetAndT2IAdapterModels = buildModelsHook(isControlNetOrT2IAdapterModelConfig);
export const useControlNetModels = buildModelsHook(isControlNetModelConfig);
export const useT2IAdapterModels = buildModelsHook(isT2IAdapterModelConfig);
export const useT5EncoderModels = buildModelsHook(isT5EncoderModelConfig);
export const useCLIPEmbedModels = buildModelsHook(isCLIPEmbedModelConfig);
export const useSpandrelImageToImageModels = buildModelsHook(isSpandrelImageToImageModelConfig);
export const useIPAdapterModels = buildModelsHook(isIPAdapterModelConfig);
export const useEmbeddingModels = buildModelsHook(isTIModelConfig);
export const useVAEModels = buildModelsHook(isVAEModelConfig);
export const useFluxVAEModels = buildModelsHook(isFluxVAEModelConfig);
export const useCLIPVisionModels = buildModelsHook(isCLIPVisionModelConfig);

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
