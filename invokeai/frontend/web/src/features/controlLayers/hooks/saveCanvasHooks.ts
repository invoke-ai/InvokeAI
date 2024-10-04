import { logger } from 'app/logging/logger';
import { useAppDispatch, useAppSelector, useAppStore } from 'app/store/storeHooks';
import type { SerializableObject } from 'common/types';
import { deepClone } from 'common/util/deepClone';
import { withResultAsync } from 'common/util/result';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectDefaultControlAdapter, selectDefaultIPAdapter } from 'features/controlLayers/hooks/addLayerHooks';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import {
  controlLayerAdded,
  entityRasterized,
  rasterLayerAdded,
  referenceImageAdded,
  referenceImageIPAdapterImageChanged,
  rgAdded,
  rgIPAdapterImageChanged,
} from 'features/controlLayers/store/canvasSlice';
import { selectCanvasMetadata } from 'features/controlLayers/store/selectors';
import type {
  CanvasControlLayerState,
  CanvasEntityIdentifier,
  CanvasRasterLayerState,
  CanvasReferenceImageState,
  CanvasRegionalGuidanceState,
  Rect,
  RegionalGuidanceReferenceImageState,
} from 'features/controlLayers/store/types';
import { imageDTOToImageObject, imageDTOToImageWithDims } from 'features/controlLayers/store/util';
import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { serializeError } from 'serialize-error';
import type { ImageDTO } from 'services/api/types';

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

  const canvasManager = useCanvasManager();

  const saveCanvas = useCallback(async () => {
    const rect =
      region === 'bbox' ? canvasManager.stateApi.getBbox().rect : canvasManager.stage.getVisibleRect('raster_layer');

    if (rect.width === 0 || rect.height === 0) {
      toast({
        title: toastError,
        description: t('controlLayers.regionIsEmpty'),
        status: 'warning',
      });
      return;
    }

    let metadata: SerializableObject | undefined = undefined;

    if (withMetadata) {
      metadata = selectCanvasMetadata(store.getState());
    }

    const result = await withResultAsync(() =>
      canvasManager.compositor.rasterizeAndUploadCompositeRasterLayer(rect, {
        is_intermediate: !saveToGallery,
        metadata,
      })
    );

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
    canvasManager.stage,
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
  const dispatch = useAppDispatch();
  const defaultIPAdapter = useAppSelector(selectDefaultIPAdapter);

  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO) => {
      const ipAdapter: RegionalGuidanceReferenceImageState = {
        id: getPrefixedId('regional_guidance_reference_image'),
        ipAdapter: {
          ...deepClone(defaultIPAdapter),
          image: imageDTOToImageWithDims(imageDTO),
        },
      };
      const overrides: Partial<CanvasRegionalGuidanceState> = {
        referenceImages: [ipAdapter],
      };

      dispatch(rgAdded({ overrides, isSelected: true }));
    };

    return {
      region: 'bbox',
      saveToGallery: false,
      onSave,
      toastOk: t('controlLayers.newRegionalReferenceImageOk'),
      toastError: t('controlLayers.newRegionalReferenceImageError'),
    };
  }, [defaultIPAdapter, dispatch, t]);
  const func = useSaveCanvas(arg);
  return func;
};

export const useNewGlobalReferenceImageFromBbox = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const defaultIPAdapter = useAppSelector(selectDefaultIPAdapter);

  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO) => {
      const overrides: Partial<CanvasReferenceImageState> = {
        ipAdapter: {
          ...deepClone(defaultIPAdapter),
          image: imageDTOToImageWithDims(imageDTO),
        },
      };
      dispatch(referenceImageAdded({ overrides, isSelected: true }));
    };

    return {
      region: 'bbox',
      saveToGallery: false,
      onSave,
      toastOk: t('controlLayers.newGlobalReferenceImageOk'),
      toastError: t('controlLayers.newGlobalReferenceImageError'),
    };
  }, [defaultIPAdapter, dispatch, t]);
  const func = useSaveCanvas(arg);
  return func;
};

export const useNewRasterLayerFromBbox = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO, rect: Rect) => {
      const overrides: Partial<CanvasRasterLayerState> = {
        objects: [imageDTOToImageObject(imageDTO)],
        position: { x: rect.x, y: rect.y },
      };
      dispatch(rasterLayerAdded({ overrides, isSelected: true }));
    };

    return {
      region: 'bbox',
      saveToGallery: false,
      onSave,
      toastOk: t('controlLayers.newRasterLayerOk'),
      toastError: t('controlLayers.newRasterLayerError'),
    };
  }, [dispatch, t]);
  const newRasterLayerFromBbox = useSaveCanvas(arg);
  return newRasterLayerFromBbox;
};

export const useNewControlLayerFromBbox = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const defaultControlAdapter = useAppSelector(selectDefaultControlAdapter);

  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO, rect: Rect) => {
      const overrides: Partial<CanvasControlLayerState> = {
        objects: [imageDTOToImageObject(imageDTO)],
        controlAdapter: deepClone(defaultControlAdapter),
        position: { x: rect.x, y: rect.y },
      };
      dispatch(controlLayerAdded({ overrides, isSelected: true }));
    };

    return {
      region: 'bbox',
      saveToGallery: false,
      onSave,
      toastOk: t('controlLayers.newControlLayerOk'),
      toastError: t('controlLayers.newControlLayerError'),
    };
  }, [defaultControlAdapter, dispatch, t]);
  const func = useSaveCanvas(arg);
  return func;
};

export const usePullBboxIntoLayer = (entityIdentifier: CanvasEntityIdentifier<'control_layer' | 'raster_layer'>) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO, rect: Rect) => {
      dispatch(
        entityRasterized({
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

export const usePullBboxIntoGlobalReferenceImage = (entityIdentifier: CanvasEntityIdentifier<'reference_image'>) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO, _: Rect) => {
      dispatch(referenceImageIPAdapterImageChanged({ entityIdentifier, imageDTO }));
    };

    return {
      region: 'bbox',
      saveToGallery: false,
      onSave,
      toastOk: t('controlLayers.pullBboxIntoReferenceImageOk'),
      toastError: t('controlLayers.pullBboxIntoReferenceImageError'),
    };
  }, [dispatch, entityIdentifier, t]);

  const func = useSaveCanvas(arg);
  return func;
};

export const usePullBboxIntoRegionalGuidanceReferenceImage = (
  entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>,
  referenceImageId: string
) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO, _: Rect) => {
      dispatch(rgIPAdapterImageChanged({ entityIdentifier, referenceImageId, imageDTO }));
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
