import { logger } from 'app/logging/logger';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { withResultAsync } from 'common/util/result';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectDefaultControlAdapter, selectDefaultIPAdapter } from 'features/controlLayers/hooks/addLayerHooks';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import {
  controlLayerAdded,
  entityRasterized,
  ipaAdded,
  ipaImageChanged,
  rasterLayerAdded,
  rgAdded,
  rgIPAdapterImageChanged,
} from 'features/controlLayers/store/canvasSlice';
import type {
  CanvasControlLayerState,
  CanvasEntityIdentifier,
  CanvasIPAdapterState,
  CanvasRasterLayerState,
  CanvasRegionalGuidanceState,
  Rect,
  RegionalGuidanceIPAdapterConfig,
} from 'features/controlLayers/store/types';
import { imageDTOToImageObject, imageDTOToImageWithDims } from 'features/controlLayers/store/types';
import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { serializeError } from 'serialize-error';
import type { ImageDTO } from 'services/api/types';

const log = logger('canvas');

type UseSaveCanvasArg = {
  region: 'canvas' | 'bbox';
  saveToGallery: boolean;
  onSave?: (imageDTO: ImageDTO, rect: Rect) => void;
};

const useSaveCanvas = ({ region, saveToGallery, onSave }: UseSaveCanvasArg) => {
  const { t } = useTranslation();

  const canvasManager = useCanvasManager();

  const saveCanvas = useCallback(async () => {
    const rect =
      region === 'bbox' ? canvasManager.stateApi.getBbox().rect : canvasManager.stage.getVisibleRect('raster_layer');

    if (rect.width === 0 || rect.height === 0) {
      toast({
        title: t('controlLayers.savedToGalleryError'),
        description: t('controlLayers.regionIsEmpty'),
        status: 'warning',
      });
      return;
    }

    const result = await withResultAsync(() =>
      canvasManager.compositor.rasterizeAndUploadCompositeRasterLayer(rect, saveToGallery)
    );

    if (result.isOk()) {
      if (onSave) {
        onSave(result.value, rect);
      }
      toast({ title: t('controlLayers.savedToGalleryOk') });
    } else {
      log.error({ error: serializeError(result.error) }, 'Failed to save canvas to gallery');
      toast({ title: t('controlLayers.savedToGalleryError'), status: 'error' });
    }
  }, [canvasManager.compositor, canvasManager.stage, canvasManager.stateApi, onSave, region, saveToGallery, t]);

  return saveCanvas;
};

const saveCanvasToGalleryArg: UseSaveCanvasArg = { region: 'canvas', saveToGallery: true };
export const useSaveCanvasToGallery = () => {
  const saveCanvasToGallery = useSaveCanvas(saveCanvasToGalleryArg);
  return saveCanvasToGallery;
};

const saveBboxToGalleryArg: UseSaveCanvasArg = { region: 'bbox', saveToGallery: true };
export const useSaveBboxToGallery = () => {
  const saveBboxToGallery = useSaveCanvas(saveBboxToGalleryArg);
  return saveBboxToGallery;
};

export const useNewRegionalIPAdapterFromBbox = () => {
  const dispatch = useAppDispatch();
  const defaultIPAdapter = useAppSelector(selectDefaultIPAdapter);

  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO) => {
      const ipAdapter: RegionalGuidanceIPAdapterConfig = {
        ...defaultIPAdapter,
        id: getPrefixedId('regional_guidance_ip_adapter'),
        image: imageDTOToImageWithDims(imageDTO),
      };
      const overrides: Partial<CanvasRegionalGuidanceState> = {
        ipAdapters: [ipAdapter],
      };

      dispatch(rgAdded({ overrides, isSelected: true }));
    };

    return { region: 'bbox', saveToGallery: false, onSave };
  }, [defaultIPAdapter, dispatch]);
  const newRegionalIPAdapterFromBbox = useSaveCanvas(arg);
  return newRegionalIPAdapterFromBbox;
};

export const useNewGlobalIPAdapterFromBbox = () => {
  const dispatch = useAppDispatch();
  const defaultIPAdapter = useAppSelector(selectDefaultIPAdapter);

  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO) => {
      const overrides: Partial<CanvasIPAdapterState> = {
        ipAdapter: {
          ...defaultIPAdapter,
          image: imageDTOToImageWithDims(imageDTO),
        },
      };
      dispatch(ipaAdded({ overrides, isSelected: true }));
    };

    return { region: 'bbox', saveToGallery: false, onSave };
  }, [defaultIPAdapter, dispatch]);
  const newGlobalIPAdapterFromBbox = useSaveCanvas(arg);
  return newGlobalIPAdapterFromBbox;
};

export const useNewRasterLayerFromBbox = () => {
  const dispatch = useAppDispatch();
  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO, rect: Rect) => {
      const overrides: Partial<CanvasRasterLayerState> = {
        objects: [imageDTOToImageObject(imageDTO)],
        position: { x: rect.x, y: rect.y },
      };
      dispatch(rasterLayerAdded({ overrides, isSelected: true }));
    };

    return { region: 'bbox', saveToGallery: false, onSave };
  }, [dispatch]);
  const newRasterLayerFromBbox = useSaveCanvas(arg);
  return newRasterLayerFromBbox;
};

export const useNewControlLayerFromBbox = () => {
  const dispatch = useAppDispatch();
  const defaultControlAdapter = useAppSelector(selectDefaultControlAdapter);

  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO, rect: Rect) => {
      const overrides: Partial<CanvasControlLayerState> = {
        objects: [imageDTOToImageObject(imageDTO)],
        controlAdapter: defaultControlAdapter,
        position: { x: rect.x, y: rect.y },
      };
      dispatch(controlLayerAdded({ overrides, isSelected: true }));
    };

    return { region: 'bbox', saveToGallery: false, onSave };
  }, [defaultControlAdapter, dispatch]);
  const newControlLayerFromBbox = useSaveCanvas(arg);
  return newControlLayerFromBbox;
};

export const usePullBboxIntoLayer = (entityIdentifier: CanvasEntityIdentifier<'control_layer' | 'raster_layer'>) => {
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

    return { region: 'bbox', saveToGallery: false, onSave };
  }, [dispatch, entityIdentifier]);

  const pullBboxIntoLayer = useSaveCanvas(arg);
  return pullBboxIntoLayer;
};

export const usePullBboxIntoIPAdapter = (entityIdentifier: CanvasEntityIdentifier<'ip_adapter'>) => {
  const dispatch = useAppDispatch();

  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO, _: Rect) => {
      dispatch(ipaImageChanged({ entityIdentifier, imageDTO }));
    };

    return { region: 'bbox', saveToGallery: false, onSave };
  }, [dispatch, entityIdentifier]);

  const pullBboxIntoIPAdapter = useSaveCanvas(arg);
  return pullBboxIntoIPAdapter;
};

export const usePullBboxIntoRegionalGuidanceIPAdapter = (
  entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>,
  ipAdapterId: string
) => {
  const dispatch = useAppDispatch();

  const arg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO, _: Rect) => {
      dispatch(rgIPAdapterImageChanged({ entityIdentifier, ipAdapterId, imageDTO }));
    };

    return { region: 'bbox', saveToGallery: false, onSave };
  }, [dispatch, entityIdentifier, ipAdapterId]);

  const pullBboxIntoRegionalGuidanceIPAdapter = useSaveCanvas(arg);
  return pullBboxIntoRegionalGuidanceIPAdapter;
};
