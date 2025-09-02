import { logger } from 'app/logging/logger';
import { useAppDispatch, useAppStore } from 'app/store/storeHooks';
import { deepClone } from 'common/util/deepClone';
import { withResultAsync } from 'common/util/result';
import { useCanvasContext } from 'features/controlLayers/contexts/CanvasInstanceContext';
import {
  getDefaultRefImageConfig,
  getDefaultRegionalGuidanceRefImageConfig,
} from 'features/controlLayers/hooks/addLayerHooks';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import {
  instanceActions,
} from 'features/controlLayers/store/canvasInstanceSlice';
import {
  selectMainModelConfig,
  selectNegativePrompt,
  selectPositivePrompt,
  selectSeed,
} from 'features/controlLayers/store/paramsSlice';
import { refImageAdded, refImageImageChanged } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasMetadata } from 'features/controlLayers/store/selectors';
import type {
  CanvasControlLayerState,
  CanvasEntityIdentifier,
  CanvasRasterLayerState,
  CanvasRegionalGuidanceState,
  Rect,
  RefImageState,
  RegionalGuidanceRefImageState,
} from 'features/controlLayers/store/types';
import { imageDTOToImageObject, imageDTOToImageWithDims, initialControlNet } from 'features/controlLayers/store/util';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import type { BoardId } from 'features/gallery/store/types';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { serializeError } from 'serialize-error';
import type { ImageDTO } from 'services/api/types';
import type { JsonObject } from 'type-fest';

const log = logger('canvas');

type UseSaveCanvasArg = {
  region: 'canvas' | 'bbox';
  saveToGallery: boolean;
  toastOk: string;
  toastError: string;
  onSave?: (imageDTO: ImageDTO, rect: Rect) => void;
  withMetadata?: boolean;
};

const useSaveCanvas = ({ region, saveToGallery, toastOk, toastError, onSave, withMetadata }: UseSaveCanvasArg) => {
  const { t } = useTranslation();
  const store = useAppStore();
  const { manager: canvasManager } = useCanvasContext();

  const saveCanvas = useCallback(async () => {
    const bbox = canvasManager.stateApi.getBbox();
    if (!bbox) {
      toast({
        title: toastError,
        description: t('controlLayers.regionIsEmpty'),
        status: 'error',
      });
      return;
    }
    
    const rect =
      region === 'bbox'
        ? bbox.rect
        : canvasManager.compositor.getVisibleRectOfType('raster_layer');

    if (rect.width === 0 || rect.height === 0) {
      toast({
        title: toastError,
        description: t('controlLayers.regionIsEmpty'),
        status: 'warning',
      });
      return;
    }

    let metadata: JsonObject | undefined = undefined;

    const state = store.getState();

    if (withMetadata) {
      metadata = selectCanvasMetadata(state);
      metadata.positive_prompt = selectPositivePrompt(state);
      metadata.negative_prompt = selectNegativePrompt(state);
      metadata.seed = selectSeed(state);
      const model = selectMainModelConfig(state);
      if (model) {
        metadata.model = Graph.getModelMetadataField(model);
      }
    }

    let boardId: BoardId | undefined = undefined;
    if (saveToGallery) {
      boardId = selectAutoAddBoardId(state);
    }

    const result = await withResultAsync(() => {
      const rasterAdapters = canvasManager.compositor.getVisibleAdaptersOfType('raster_layer');
      return canvasManager.compositor.getCompositeImageDTO(
        rasterAdapters,
        rect,
        {
          is_intermediate: !saveToGallery,
          metadata,
          board_id: boardId,
          silent: true,
        },
        undefined,
        true // force upload the image to ensure it gets added to the gallery
      );
    });

    if (result.isOk()) {
      if (onSave) {
        onSave(result.value, rect);
      }
      toast({ title: toastOk });
    } else {
      log.error({ error: serializeError(result.error) }, 'Failed to save canvas to gallery');
      toast({ title: toastError, status: 'error' });
    }
  }, [
    canvasManager.compositor,
    canvasManager.stateApi,
    onSave,
    region,
    saveToGallery,
    store,
    t,
    toastError,
    toastOk,
    withMetadata,
  ]);

  return saveCanvas;
};

export const useSaveCanvasToGallery = () => {
  const { t } = useTranslation();
  const arg: UseSaveCanvasArg = useMemo(
    () => ({
      region: 'canvas',
      saveToGallery: true,
      toastOk: t('controlLayers.savedToGalleryOk'),
      toastError: t('controlLayers.savedToGalleryError'),
      withMetadata: true,
    }),
    [t]
  );
  const func = useSaveCanvas(arg);
  return func;
};

export const useSaveBboxToGallery = () => {
  const { t } = useTranslation();
  const arg: UseSaveCanvasArg = useMemo(
    () => ({
      region: 'bbox',
      saveToGallery: true,
      toastOk: t('controlLayers.savedToGalleryOk'),
      toastError: t('controlLayers.savedToGalleryError'),
      withMetadata: true,
    }),
    [t]
  );
  const func = useSaveCanvas(arg);
  return func;
};

export const useNewRegionalReferenceImageFromBbox = () => {
  const { t } = useTranslation();
  const { dispatch, getState } = useAppStore();

  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO) => {
      const ipAdapter: RegionalGuidanceRefImageState = {
        id: getPrefixedId('regional_guidance_reference_image'),
        config: {
          ...getDefaultRegionalGuidanceRefImageConfig(getState),
          image: imageDTOToImageWithDims(imageDTO),
        },
      };
      const overrides: Partial<CanvasRegionalGuidanceState> = {
        referenceImages: [ipAdapter],
      };

      dispatch(instanceActions.rgAdded({ overrides, isSelected: true }));
    };

    return {
      region: 'bbox',
      saveToGallery: false,
      onSave,
      toastOk: t('controlLayers.newRegionalReferenceImageOk'),
      toastError: t('controlLayers.newRegionalReferenceImageError'),
    };
  }, [dispatch, getState, t]);
  const func = useSaveCanvas(arg);
  return func;
};

export const useNewGlobalReferenceImageFromBbox = () => {
  const { t } = useTranslation();
  const { dispatch, getState } = useAppStore();

  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO) => {
      const overrides: Partial<RefImageState> = {
        config: {
          ...getDefaultRefImageConfig(getState),
          image: imageDTOToImageWithDims(imageDTO),
        },
      };
      dispatch(refImageAdded({ overrides }));
    };

    return {
      region: 'bbox',
      saveToGallery: false,
      onSave,
      toastOk: t('controlLayers.newGlobalReferenceImageOk'),
      toastError: t('controlLayers.newGlobalReferenceImageError'),
    };
  }, [dispatch, getState, t]);
  const func = useSaveCanvas(arg);
  return func;
};

export const useNewRasterLayerFromBbox = () => {
  const { t } = useTranslation();
  const { dispatch } = useCanvasContext();
  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO, rect: Rect) => {
      const overrides: Partial<CanvasRasterLayerState> = {
        objects: [imageDTOToImageObject(imageDTO)],
        position: { x: rect.x, y: rect.y },
      };
      dispatch(instanceActions.rasterLayerAdded({ overrides, isSelected: true }));
    };

    return {
      region: 'bbox',
      saveToGallery: false,
      onSave,
      toastOk: t('controlLayers.newRasterLayerOk'),
      toastError: t('controlLayers.newRasterLayerError'),
    };
  }, [dispatch, t]);
  const func = useSaveCanvas(arg);
  return func;
};

export const useNewControlLayerFromBbox = () => {
  const { t } = useTranslation();
  const { dispatch } = useCanvasContext();

  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO, rect: Rect) => {
      const overrides: Partial<CanvasControlLayerState> = {
        objects: [imageDTOToImageObject(imageDTO)],
        controlAdapter: deepClone(initialControlNet),
        position: { x: rect.x, y: rect.y },
      };
      dispatch(instanceActions.controlLayerAdded({ overrides, isSelected: true }));
    };

    return {
      region: 'bbox',
      saveToGallery: false,
      onSave,
      toastOk: t('controlLayers.newControlLayerOk'),
      toastError: t('controlLayers.newControlLayerError'),
    };
  }, [dispatch, t]);
  const func = useSaveCanvas(arg);
  return func;
};

export const usePullBboxIntoLayer = (entityIdentifier: CanvasEntityIdentifier<'control_layer' | 'raster_layer'>) => {
  const { t } = useTranslation();
  const { dispatch } = useCanvasContext();

  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO, rect: Rect) => {
      dispatch(
        instanceActions.entityRasterized({
          entityIdentifier,
          position: { x: rect.x, y: rect.y },
          imageObject: imageDTOToImageObject(imageDTO),
          replaceObjects: true,
        })
      );
    };

    return {
      region: 'bbox',
      saveToGallery: false,
      onSave,
      toastOk: t('controlLayers.pullBboxIntoLayerOk'),
      toastError: t('controlLayers.pullBboxIntoLayerError'),
    };
  }, [dispatch, entityIdentifier, t]);

  const func = useSaveCanvas(arg);
  return func;
};

export const usePullBboxIntoGlobalReferenceImage = (id: string) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO, _: Rect) => {
      dispatch(refImageImageChanged({ id, imageDTO }));
    };

    return {
      region: 'bbox',
      saveToGallery: false,
      onSave,
      toastOk: t('controlLayers.pullBboxIntoReferenceImageOk'),
      toastError: t('controlLayers.pullBboxIntoReferenceImageError'),
    };
  }, [dispatch, id, t]);

  const func = useSaveCanvas(arg);
  return func;
};

export const usePullBboxIntoRegionalGuidanceReferenceImage = (
  entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>,
  referenceImageId: string
) => {
  const { t } = useTranslation();
  const { dispatch } = useCanvasContext();

  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO, _: Rect) => {
      dispatch(instanceActions.rgRefImageImageChanged({ entityIdentifier, referenceImageId, imageDTO }));
    };

    return {
      region: 'bbox',
      saveToGallery: false,
      onSave,
      toastOk: t('controlLayers.pullBboxIntoReferenceImageOk'),
      toastError: t('controlLayers.pullBboxIntoReferenceImageError'),
    };
  }, [dispatch, entityIdentifier, referenceImageId, t]);

  const func = useSaveCanvas(arg);
  return func;
};
