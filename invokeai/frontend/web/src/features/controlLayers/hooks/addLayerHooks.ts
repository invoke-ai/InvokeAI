import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { deepClone } from 'common/util/deepClone';
import { CanvasEntityAdapterBase } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { canvasReset } from 'features/controlLayers/store/actions';
import {
  bboxChangedFromCanvas,
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
import {
  selectBboxModelBase,
  selectBboxRect,
  selectCanvasSlice,
  selectEntityOrThrow,
} from 'features/controlLayers/store/selectors';
import type {
  CanvasEntityIdentifier,
  CanvasRasterLayerState,
  CanvasRegionalGuidanceState,
  ControlNetConfig,
  IPAdapterConfig,
  T2IAdapterConfig,
} from 'features/controlLayers/store/types';
import {
  imageDTOToImageObject,
  initialControlNet,
  initialIPAdapter,
  initialT2IAdapter,
} from 'features/controlLayers/store/util';
import { calculateNewSize } from 'features/controlLayers/util/getScaledBoundingBoxDimensions';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { getOptimalDimension } from 'features/parameters/util/optimalDimension';
import { useCallback } from 'react';
import { modelConfigsAdapterSelectors, selectModelConfigsQuery } from 'services/api/endpoints/models';
import type { ControlNetModelConfig, ImageDTO, IPAdapterModelConfig, T2IAdapterModelConfig } from 'services/api/types';
import { isControlNetOrT2IAdapterModelConfig, isIPAdapterModelConfig } from 'services/api/types';

export const selectDefaultControlAdapter = createSelector(
  selectModelConfigsQuery,
  selectBase,
  (query, base): ControlNetConfig | T2IAdapterConfig => {
    const { data } = query;
    let model: ControlNetModelConfig | T2IAdapterModelConfig | null = null;
    if (data) {
      const modelConfigs = modelConfigsAdapterSelectors
        .selectAll(data)
        .filter(isControlNetOrT2IAdapterModelConfig)
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
    }
    return ipAdapter;
  }
);

export const useAddControlLayer = () => {
  const dispatch = useAppDispatch();
  const defaultControlAdapter = useAppSelector(selectDefaultControlAdapter);
  const func = useCallback(() => {
    const overrides = { controlAdapter: defaultControlAdapter };
    dispatch(controlLayerAdded({ isSelected: true, overrides }));
  }, [defaultControlAdapter, dispatch]);

  return func;
};

export const useAddRasterLayer = () => {
  const dispatch = useAppDispatch();
  const func = useCallback(() => {
    dispatch(rasterLayerAdded({ isSelected: true }));
  }, [dispatch]);

  return func;
};

export const useNewRasterLayerFromImage = () => {
  const dispatch = useAppDispatch();
  const bboxRect = useAppSelector(selectBboxRect);
  const func = useCallback(
    (imageDTO: ImageDTO) => {
      const imageObject = imageDTOToImageObject(imageDTO);
      const overrides: Partial<CanvasRasterLayerState> = {
        position: { x: bboxRect.x, y: bboxRect.y },
        objects: [imageObject],
      };
      dispatch(rasterLayerAdded({ overrides, isSelected: true }));
    },
    [bboxRect.x, bboxRect.y, dispatch]
  );

  return func;
};

/**
 * Returns a function that adds a new canvas with the given image as the initial image, replicating the img2img flow:
 * - Reset the canvas
 * - Resize the bbox to the image's aspect ratio at the optimal size for the selected model
 * - Add the image as a raster layer
 * - Resizes the layer to fit the bbox using the 'fill' strategy
 *
 * This allows the user to immediately generate a new image from the given image without any additional steps.
 */
export const useNewCanvasFromImage = () => {
  const dispatch = useAppDispatch();
  const bboxRect = useAppSelector(selectBboxRect);
  const base = useAppSelector(selectBboxModelBase);
  const func = useCallback(
    (imageDTO: ImageDTO) => {
      // Calculate the new bbox dimensions to fit the image's aspect ratio at the optimal size
      const ratio = imageDTO.width / imageDTO.height;
      const optimalDimension = getOptimalDimension(base);
      const { width, height } = calculateNewSize(ratio, optimalDimension ** 2, base);

      // The overrides need to include the layer's ID so we can transform the layer it is initialized
      const overrides = {
        id: getPrefixedId('raster_layer'),
        position: { x: bboxRect.x, y: bboxRect.y },
        objects: [imageDTOToImageObject(imageDTO)],
      } satisfies Partial<CanvasRasterLayerState>;

      CanvasEntityAdapterBase.registerInitCallback(async (adapter) => {
        // Skip the callback if the adapter is not the one we are creating
        if (adapter.id !== overrides.id) {
          return false;
        }
        // Fit the layer to the bbox w/ fill strategy
        await adapter.transformer.startTransform({ silent: true });
        adapter.transformer.fitToBboxFill();
        await adapter.transformer.applyTransform();
        return true;
      });

      dispatch(canvasReset());
      // The `bboxChangedFromCanvas` reducer does no validation! Careful!
      dispatch(bboxChangedFromCanvas({ x: 0, y: 0, width, height }));
      dispatch(rasterLayerAdded({ overrides, isSelected: true }));
    },
    [base, bboxRect.x, bboxRect.y, dispatch]
  );

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
      referenceImages: [{ id: getPrefixedId('regional_guidance_reference_image'), ipAdapter: defaultIPAdapter }],
    };
    dispatch(rgAdded({ isSelected: true, overrides }));
  }, [defaultIPAdapter, dispatch]);

  return func;
};

export const useAddGlobalReferenceImage = () => {
  const dispatch = useAppDispatch();
  const defaultIPAdapter = useAppSelector(selectDefaultIPAdapter);
  const func = useCallback(() => {
    const overrides = { ipAdapter: defaultIPAdapter };
    dispatch(referenceImageAdded({ isSelected: true, overrides }));
  }, [defaultIPAdapter, dispatch]);

  return func;
};

export const useAddRegionalGuidanceIPAdapter = (entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>) => {
  const dispatch = useAppDispatch();
  const defaultIPAdapter = useAppSelector(selectDefaultIPAdapter);
  const func = useCallback(() => {
    dispatch(rgIPAdapterAdded({ entityIdentifier, overrides: { ipAdapter: defaultIPAdapter } }));
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
    const entity = selectEntityOrThrow(canvas, entityIdentifier);
    return {
      canAddPositivePrompt: entity?.positivePrompt === null,
      canAddNegativePrompt: entity?.negativePrompt === null,
    };
  });
};
