import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppStore } from 'app/store/nanostores/store';
import type { AppGetState } from 'app/store/store';
import { useAppDispatch } from 'app/store/storeHooks';
import { deepClone } from 'common/util/deepClone';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import {
  controlLayerAdded,
  inpaintMaskAdded,
  inpaintMaskDenoiseLimitAdded,
  inpaintMaskNoiseAdded,
  rasterLayerAdded,
  rgAdded,
  rgNegativePromptChanged,
  rgPositivePromptChanged,
  rgRefImageAdded,
} from 'features/controlLayers/store/canvasSlice';
import { selectBase, selectMainModelConfig } from 'features/controlLayers/store/paramsSlice';
import { refImageAdded } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasSlice, selectEntity } from 'features/controlLayers/store/selectors';
import type {
  CanvasEntityIdentifier,
  CanvasRegionalGuidanceState,
  ChatGPT4oReferenceImageConfig,
  ControlLoRAConfig,
  ControlNetConfig,
  FluxKontextReferenceImageConfig,
  IPAdapterConfig,
  T2IAdapterConfig,
} from 'features/controlLayers/store/types';
import {
  initialChatGPT4oReferenceImage,
  initialControlNet,
  initialFluxKontextReferenceImage,
  initialIPAdapter,
  initialT2IAdapter,
} from 'features/controlLayers/store/util';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { useCallback } from 'react';
import { modelConfigsAdapterSelectors, selectModelConfigsQuery } from 'services/api/endpoints/models';
import { selectIPAdapterModels } from 'services/api/hooks/modelsByType';
import type { ControlLoRAModelConfig, ControlNetModelConfig, T2IAdapterModelConfig } from 'services/api/types';
import { isControlLayerModelConfig } from 'services/api/types';

/**
 * Selects the default control adapter configuration based on the model configurations and the base.
 *
 * Be sure to clone the output of this selector before modifying it!
 *
 * @knipignore
 */
export const selectDefaultControlAdapter = createSelector(
  selectModelConfigsQuery,
  selectBase,
  (query, base): ControlNetConfig | T2IAdapterConfig | ControlLoRAConfig => {
    const { data } = query;
    let model: ControlNetModelConfig | T2IAdapterModelConfig | ControlLoRAModelConfig | null = null;
    if (data) {
      const modelConfigs = modelConfigsAdapterSelectors
        .selectAll(data)
        .filter(isControlLayerModelConfig)
        .sort((a) => (a.type === 'controlnet' ? -1 : 1)); // Prefer ControlNet models
      const compatibleModels = modelConfigs.filter((m) => (base ? m.base === base : true));
      model = compatibleModels[0] ?? modelConfigs[0] ?? null;
    }
    const controlAdapter = model?.type === 't2i_adapter' ? deepClone(initialT2IAdapter) : deepClone(initialControlNet);
    if (model) {
      controlAdapter.model = zModelIdentifierField.parse(model);
    }
    return controlAdapter;
  }
);

export const getDefaultRefImageConfig = (
  getState: AppGetState
): IPAdapterConfig | ChatGPT4oReferenceImageConfig | FluxKontextReferenceImageConfig => {
  const state = getState();

  const mainModelConfig = selectMainModelConfig(state);
  const ipAdapterModelConfigs = selectIPAdapterModels(state);

  const base = mainModelConfig?.base;

  // For ChatGPT-4o, the ref image model is the model itself.
  if (base === 'chatgpt-4o') {
    const config = deepClone(initialChatGPT4oReferenceImage);
    config.model = zModelIdentifierField.parse(mainModelConfig);
    return config;
  }

  if (base === 'flux-kontext') {
    const config = deepClone(initialFluxKontextReferenceImage);
    config.model = zModelIdentifierField.parse(mainModelConfig);
    return config;
  }

  // Otherwise, find the first compatible IP Adapter model.
  const modelConfig = ipAdapterModelConfigs.find((m) => m.base === base);

  // Clone the initial IP Adapter config and set the model if available.
  const config = deepClone(initialIPAdapter);

  if (modelConfig) {
    config.model = zModelIdentifierField.parse(modelConfig);
    // FLUX models use a different vision model.
    if (modelConfig.base === 'flux') {
      config.clipVisionModel = 'ViT-L';
    }
  }
  return config;
};

export const getDefaultRegionalGuidanceRefImageConfig = (getState: AppGetState): IPAdapterConfig => {
  // Regional guidance ref images do not support ChatGPT-4o, so we always return the IP Adapter config.
  const state = getState();

  const mainModelConfig = selectMainModelConfig(state);
  const ipAdapterModelConfigs = selectIPAdapterModels(state);

  const base = mainModelConfig?.base;

  // Find the first compatible IP Adapter model.
  const modelConfig = ipAdapterModelConfigs.find((m) => m.base === base);

  // Clone the initial IP Adapter config and set the model if available.
  const config = deepClone(initialIPAdapter);

  if (modelConfig) {
    config.model = zModelIdentifierField.parse(modelConfig);
    // FLUX models use a different vision model.
    if (modelConfig.base === 'flux') {
      config.clipVisionModel = 'ViT-L';
    }
  }
  return config;
};

export const useAddControlLayer = () => {
  const dispatch = useAppDispatch();
  const func = useCallback(() => {
    const overrides = { controlAdapter: deepClone(initialControlNet) };
    dispatch(controlLayerAdded({ isSelected: true, overrides }));
  }, [dispatch]);

  return func;
};

export const useAddRasterLayer = () => {
  const dispatch = useAppDispatch();
  const func = useCallback(() => {
    dispatch(rasterLayerAdded({ isSelected: true }));
  }, [dispatch]);

  return func;
};

export const useAddInpaintMask = () => {
  const dispatch = useAppDispatch();
  const func = useCallback(() => {
    dispatch(inpaintMaskAdded({ isSelected: true }));
  }, [dispatch]);

  return func;
};

export const useAddRegionalGuidance = () => {
  const dispatch = useAppDispatch();
  const func = useCallback(() => {
    dispatch(rgAdded({ isSelected: true }));
  }, [dispatch]);

  return func;
};

export const useAddNewRegionalGuidanceWithARefImage = () => {
  const { dispatch, getState } = useAppStore();

  const func = useCallback(() => {
    const config = getDefaultRegionalGuidanceRefImageConfig(getState);
    const overrides: Partial<CanvasRegionalGuidanceState> = {
      referenceImages: [{ id: getPrefixedId('regional_guidance_reference_image'), config }],
    };
    dispatch(rgAdded({ isSelected: true, overrides }));
  }, [dispatch, getState]);

  return func;
};

export const useAddGlobalReferenceImage = () => {
  const { dispatch, getState } = useAppStore();
  const func = useCallback(() => {
    const config = getDefaultRefImageConfig(getState);
    const overrides = { config };
    dispatch(refImageAdded({ isSelected: true, overrides }));
  }, [dispatch, getState]);

  return func;
};

export const useAddRefImageToExistingRegionalGuidance = (
  entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>
) => {
  const { dispatch, getState } = useAppStore();
  const func = useCallback(() => {
    const config = getDefaultRegionalGuidanceRefImageConfig(getState);
    dispatch(rgRefImageAdded({ entityIdentifier, overrides: { config } }));
  }, [dispatch, entityIdentifier, getState]);

  return func;
};

export const useAddPositivePromptToExistingRegionalGuidance = (
  entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>
) => {
  const dispatch = useAppDispatch();
  const func = useCallback(() => {
    dispatch(rgPositivePromptChanged({ entityIdentifier, prompt: '' }));
  }, [dispatch, entityIdentifier]);

  return func;
};

export const useAddNegativePromptToExistingRegionalGuidance = (
  entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>
) => {
  const dispatch = useAppDispatch();
  const runc = useCallback(() => {
    dispatch(rgNegativePromptChanged({ entityIdentifier, prompt: '' }));
  }, [dispatch, entityIdentifier]);

  return runc;
};

export const useAddInpaintMaskNoise = (entityIdentifier: CanvasEntityIdentifier<'inpaint_mask'>) => {
  const dispatch = useAppDispatch();
  const func = useCallback(() => {
    dispatch(inpaintMaskNoiseAdded({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return func;
};

export const useAddInpaintMaskDenoiseLimit = (entityIdentifier: CanvasEntityIdentifier<'inpaint_mask'>) => {
  const dispatch = useAppDispatch();
  const func = useCallback(() => {
    dispatch(inpaintMaskDenoiseLimitAdded({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return func;
};

export const buildSelectValidRegionalGuidanceActions = (
  entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>
) => {
  return createMemoizedSelector(selectCanvasSlice, (canvas) => {
    const entity = selectEntity(canvas, entityIdentifier);
    return {
      canAddPositivePrompt: entity?.positivePrompt === null,
      canAddNegativePrompt: entity?.negativePrompt === null,
    };
  });
};
