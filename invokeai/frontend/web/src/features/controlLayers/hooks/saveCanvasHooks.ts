import { logger } from 'app/logging/logger';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import { isOk, withResultAsync } from 'common/util/result';
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

export const [useIsSavingCanvas] = buildUseBoolean(false);

type UseSaveCanvasArg = {
  region: 'canvas' | 'bbox';
  saveToGallery: boolean;
  onSave?: (imageDTO: ImageDTO, rect: Rect) => void;
};

const useSaveCanvas = ({ region, saveToGallery, onSave }: UseSaveCanvasArg) => {
  const { t } = useTranslation();

  const canvasManager = useCanvasManager();
  const isSaving = useIsSavingCanvas();

  const saveCanvas = useCallback(async () => {
    isSaving.setTrue();

    const rect =
      region === 'bbox' ? canvasManager.stateApi.getBbox().rect : canvasManager.stage.getVisibleRect('raster_layer');

    const result = await withResultAsync(() =>
      canvasManager.compositor.rasterizeAndUploadCompositeRasterLayer(rect, saveToGallery)
    );

    if (isOk(result)) {
      if (onSave) {
        onSave(result.value, rect);
      }
      toast({ title: t('controlLayers.savedToGalleryOk') });
    } else {
      log.error({ error: serializeError(result.error) }, 'Failed to save canvas to gallery');
      toast({ title: t('controlLayers.savedToGalleryError'), status: 'error' });
    }

    isSaving.setFalse();
  }, [
    canvasManager.compositor,
    canvasManager.stage,
    canvasManager.stateApi,
    isSaving,
    onSave,
    region,
    saveToGallery,
    t,
  ]);

  return saveCanvas;
};

export const useSaveCanvasToGallery = () => {
  const saveCanvasToGalleryArg = useMemo<UseSaveCanvasArg>(() => ({ region: 'canvas', saveToGallery: true }), []);
  const saveCanvasToGallery = useSaveCanvas(saveCanvasToGalleryArg);
  return saveCanvasToGallery;
};

export const useSaveBboxToGallery = () => {
  const saveBboxToGalleryArg = useMemo<UseSaveCanvasArg>(() => ({ region: 'bbox', saveToGallery: true }), []);
  const saveBboxToGallery = useSaveCanvas(saveBboxToGalleryArg);
  return saveBboxToGallery;
};

export const useSaveBboxAsRegionalGuidanceIPAdapter = () => {
  const dispatch = useAppDispatch();
  const defaultIPAdapter = useAppSelector(selectDefaultIPAdapter);

  const saveBboxAsRegionalGuidanceIPAdapterArg = useMemo<UseSaveCanvasArg>(() => {
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
  const saveBboxAsRegionalGuidanceIPAdapter = useSaveCanvas(saveBboxAsRegionalGuidanceIPAdapterArg);
  return saveBboxAsRegionalGuidanceIPAdapter;
};

export const useSaveBboxAsGlobalIPAdapter = () => {
  const dispatch = useAppDispatch();
  const defaultIPAdapter = useAppSelector(selectDefaultIPAdapter);

  const saveBboxAsIPAdapterArg = useMemo<UseSaveCanvasArg>(() => {
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
  const saveBboxAsIPAdapter = useSaveCanvas(saveBboxAsIPAdapterArg);
  return saveBboxAsIPAdapter;
};

export const useSaveBboxAsRasterLayer = () => {
  const dispatch = useAppDispatch();
  const saveBboxAsRasterLayerArg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO, rect: Rect) => {
      const overrides: Partial<CanvasRasterLayerState> = {
        objects: [imageDTOToImageObject(imageDTO)],
        position: { x: rect.x, y: rect.y },
      };
      dispatch(rasterLayerAdded({ overrides, isSelected: true }));
    };

    return { region: 'bbox', saveToGallery: false, onSave };
  }, [dispatch]);
  const saveBboxAsRasterLayer = useSaveCanvas(saveBboxAsRasterLayerArg);
  return saveBboxAsRasterLayer;
};

export const useSaveBboxAsControlLayer = () => {
  const dispatch = useAppDispatch();
  const defaultControlAdapter = useAppSelector(selectDefaultControlAdapter);

  const saveBboxAsControlLayerArg = useMemo<UseSaveCanvasArg>(() => {
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
  const saveBboxAsControlLayer = useSaveCanvas(saveBboxAsControlLayerArg);
  return saveBboxAsControlLayer;
};

export const usePullBboxIntoLayer = (entityIdentifier: CanvasEntityIdentifier<'control_layer' | 'raster_layer'>) => {
  const dispatch = useAppDispatch();

  const pullBboxIntoLayerArg = useMemo<UseSaveCanvasArg>(() => {
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

  const pullBboxIntoLayer = useSaveCanvas(pullBboxIntoLayerArg);
  return pullBboxIntoLayer;
};

export const usePullBboxIntoIPAdapter = (entityIdentifier: CanvasEntityIdentifier<'ip_adapter'>) => {
  const dispatch = useAppDispatch();

  const pullBboxIntoIPAdapterArg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO, _: Rect) => {
      dispatch(ipaImageChanged({ entityIdentifier, imageDTO }));
    };

    return { region: 'bbox', saveToGallery: false, onSave };
  }, [dispatch, entityIdentifier]);

  const pullBboxIntoIPAdapter = useSaveCanvas(pullBboxIntoIPAdapterArg);
  return pullBboxIntoIPAdapter;
};

export const usePullBboxIntoRegionalGuidanceIPAdapter = (
  entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>,
  ipAdapterId: string
) => {
  const dispatch = useAppDispatch();

  const pullBboxIntoRegionalGuidanceIPAdapterArg = useMemo<UseSaveCanvasArg>(() => {
    const onSave = (imageDTO: ImageDTO, _: Rect) => {
      dispatch(rgIPAdapterImageChanged({ entityIdentifier, ipAdapterId, imageDTO }));
    };

    return { region: 'bbox', saveToGallery: false, onSave };
  }, [dispatch, entityIdentifier, ipAdapterId]);

  const pullBboxIntoRegionalGuidanceIPAdapter = useSaveCanvas(pullBboxIntoRegionalGuidanceIPAdapterArg);
  return pullBboxIntoRegionalGuidanceIPAdapter;
};
