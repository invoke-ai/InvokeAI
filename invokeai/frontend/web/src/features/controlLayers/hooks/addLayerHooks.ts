import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { deepClone } from 'common/util/deepClone';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import {
  controlLayerAdded,
  inpaintMaskAdded,
  rasterLayerAdded,
  referenceImageAdded,
  rgAdded,
  rgIPAdapterAdded,
  rgNegativePromptChanged,
  rgPositivePromptChanged,
} from 'features/controlLayers/store/canvasSlice';
import { selectBase } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice, selectEntity } from 'features/controlLayers/store/selectors';
import type {
  CanvasEntityIdentifier,
  CanvasRegionalGuidanceState,
  ControlLoRAConfig,
  ControlNetConfig,
  IPAdapterConfig,
  T2IAdapterConfig,
} from 'features/controlLayers/store/types';
import { initialControlNet, initialIPAdapter, initialT2IAdapter } from 'features/controlLayers/store/util';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { useCallback } from 'react';
import { modelConfigsAdapterSelectors, selectModelConfigsQuery } from 'services/api/endpoints/models';
import type {
  ControlLoRAModelConfig,
  ControlNetModelConfig,
  IPAdapterModelConfig,
  T2IAdapterModelConfig,
} from 'services/api/types';
import { isControlLayerModelConfig, isIPAdapterModelConfig } from 'services/api/types';

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

/**
 * Selects the default IP adapter configuration based on the model configurations and the base.
 *
 * Be sure to clone the output of this selector before modifying it!
 */
export const selectDefaultIPAdapter = createSelector(
  selectModelConfigsQuery,
  selectBase,
  (query, base): IPAdapterConfig => {
    const { data } = query;
    let model: IPAdapterModelConfig | null = null;
    if (data) {
      const modelConfigs = modelConfigsAdapterSelectors.selectAll(data).filter(isIPAdapterModelConfig);
      const compatibleModels = modelConfigs.filter((m) => (base ? m.base === base : true));
      model = compatibleModels[0] ?? modelConfigs[0] ?? null;
    }
    const ipAdapter = deepClone(initialIPAdapter);
    if (model) {
      ipAdapter.model = zModelIdentifierField.parse(model);
      if (model.base === 'flux') {
        ipAdapter.clipVisionModel = 'ViT-L';
      }
    }
    return ipAdapter;
  }
);

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

export const useAddRegionalReferenceImage = () => {
  const dispatch = useAppDispatch();
  const defaultIPAdapter = useAppSelector(selectDefaultIPAdapter);

  const func = useCallback(() => {
    const overrides: Partial<CanvasRegionalGuidanceState> = {
      referenceImages: [
        { id: getPrefixedId('regional_guidance_reference_image'), ipAdapter: deepClone(defaultIPAdapter) },
      ],
    };
    dispatch(rgAdded({ isSelected: true, overrides }));
  }, [defaultIPAdapter, dispatch]);

  return func;
};

export const useAddGlobalReferenceImage = () => {
  const dispatch = useAppDispatch();
  const defaultIPAdapter = useAppSelector(selectDefaultIPAdapter);
  const func = useCallback(() => {
    const overrides = { ipAdapter: deepClone(defaultIPAdapter) };
    dispatch(referenceImageAdded({ isSelected: true, overrides }));
  }, [defaultIPAdapter, dispatch]);

  return func;
};

export const useAddRegionalGuidanceIPAdapter = (entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>) => {
  const dispatch = useAppDispatch();
  const defaultIPAdapter = useAppSelector(selectDefaultIPAdapter);
  const func = useCallback(() => {
    dispatch(rgIPAdapterAdded({ entityIdentifier, overrides: { ipAdapter: deepClone(defaultIPAdapter) } }));
  }, [defaultIPAdapter, dispatch, entityIdentifier]);

  return func;
};

export const useAddRegionalGuidancePositivePrompt = (entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>) => {
  const dispatch = useAppDispatch();
  const func = useCallback(() => {
    dispatch(rgPositivePromptChanged({ entityIdentifier, prompt: '' }));
  }, [dispatch, entityIdentifier]);

  return func;
};

export const useAddRegionalGuidanceNegativePrompt = (entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>) => {
  const dispatch = useAppDispatch();
  const runc = useCallback(() => {
    dispatch(rgNegativePromptChanged({ entityIdentifier, prompt: '' }));
  }, [dispatch, entityIdentifier]);

  return runc;
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
