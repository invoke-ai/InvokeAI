import { IconButton } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import type { AppDispatch } from 'app/store/store';
import { useAppDispatch } from 'app/store/storeHooks';
import { withResultAsync } from 'common/util/result';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityTypeCount } from 'features/controlLayers/hooks/useEntityTypeCount';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import {
  controlLayerAdded,
  inpaintMaskAdded,
  rasterLayerAdded,
  rgAdded,
} from 'features/controlLayers/store/canvasSlice';
import type { CanvasEntityType } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiStackBold } from 'react-icons/pi';
import { serializeError } from 'serialize-error';

const log = logger('canvas');

type Props = {
  type: CanvasEntityType;
};

const mergeRasterLayers = async (canvasManager: CanvasManager, dispatch: AppDispatch) => {
  const rect = canvasManager.stage.getVisibleRect('raster_layer');
  const adapters = canvasManager.compositor.getVisibleAdaptersOfType('raster_layer');
  const result = await withResultAsync(() =>
    canvasManager.compositor.getCompositeImageDTO(adapters, rect, { is_intermediate: true })
  );

  if (result.isErr()) {
    log.error({ error: serializeError(result.error) }, 'Failed to merge visible');
    toast({ title: t('controlLayers.mergeVisibleError'), status: 'error' });
    return;
  }

  dispatch(
    rasterLayerAdded({
      isSelected: true,
      overrides: {
        objects: [imageDTOToImageObject(result.value)],
        position: { x: Math.floor(rect.x), y: Math.floor(rect.y) },
      },
    })
  );

  toast({ title: t('controlLayers.mergeVisibleOk') });
};

const mergeInpaintMasks = async (canvasManager: CanvasManager, dispatch: AppDispatch) => {
  const rect = canvasManager.stage.getVisibleRect('inpaint_mask');
  const adapters = canvasManager.compositor.getVisibleAdaptersOfType('inpaint_mask');
  const result = await withResultAsync(() =>
    canvasManager.compositor.getCompositeImageDTO(adapters, rect, { is_intermediate: true })
  );

  if (result.isErr()) {
    log.error({ error: serializeError(result.error) }, 'Failed to merge visible');
    toast({ title: t('controlLayers.mergeVisibleError'), status: 'error' });
    return;
  }

  dispatch(
    inpaintMaskAdded({
      isSelected: true,
      overrides: {
        objects: [imageDTOToImageObject(result.value)],
        position: { x: Math.floor(rect.x), y: Math.floor(rect.y) },
      },
    })
  );

  toast({ title: t('controlLayers.mergeVisibleOk') });
};

const mergeRegionalGuidance = async (canvasManager: CanvasManager, dispatch: AppDispatch) => {
  const rect = canvasManager.stage.getVisibleRect('regional_guidance');
  const adapters = canvasManager.compositor.getVisibleAdaptersOfType('regional_guidance');
  const result = await withResultAsync(() =>
    canvasManager.compositor.getCompositeImageDTO(adapters, rect, { is_intermediate: true })
  );

  if (result.isErr()) {
    log.error({ error: serializeError(result.error) }, 'Failed to merge visible');
    toast({ title: t('controlLayers.mergeVisibleError'), status: 'error' });
    return;
  }

  dispatch(
    rgAdded({
      isSelected: true,
      overrides: {
        objects: [imageDTOToImageObject(result.value)],
        position: { x: Math.floor(rect.x), y: Math.floor(rect.y) },
      },
    })
  );

  toast({ title: t('controlLayers.mergeVisibleOk') });
};

const mergeControlLayers = async (canvasManager: CanvasManager, dispatch: AppDispatch) => {
  const rect = canvasManager.stage.getVisibleRect('control_layer');
  const adapters = canvasManager.compositor.getVisibleAdaptersOfType('control_layer');
  const result = await withResultAsync(() =>
    canvasManager.compositor.getCompositeImageDTO(
      adapters,
      rect,
      { is_intermediate: true },
      { globalCompositeOperation: 'lighter' }
    )
  );

  if (result.isErr()) {
    log.error({ error: serializeError(result.error) }, 'Failed to merge visible');
    toast({ title: t('controlLayers.mergeVisibleError'), status: 'error' });
    return;
  }

  dispatch(
    controlLayerAdded({
      isSelected: true,
      overrides: {
        objects: [imageDTOToImageObject(result.value)],
        position: { x: Math.floor(rect.x), y: Math.floor(rect.y) },
      },
    })
  );

  toast({ title: t('controlLayers.mergeVisibleOk') });
};

export const CanvasEntityMergeVisibleButton = memo(({ type }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const canvasManager = useCanvasManager();
  const isBusy = useCanvasIsBusy();
  const entityCount = useEntityTypeCount(type);
  const onClick = useCallback(() => {
    switch (type) {
      case 'raster_layer':
        mergeRasterLayers(canvasManager, dispatch);
        break;
      case 'inpaint_mask':
        mergeInpaintMasks(canvasManager, dispatch);
        break;
      case 'regional_guidance':
        mergeRegionalGuidance(canvasManager, dispatch);
        break;
      case 'control_layer':
        mergeControlLayers(canvasManager, dispatch);
        break;
      default:
        log.error({ type }, 'Unsupported type for merge visible');
    }
  }, [canvasManager, dispatch, type]);

  return (
    <IconButton
      size="sm"
      aria-label={t('controlLayers.mergeVisible')}
      tooltip={t('controlLayers.mergeVisible')}
      variant="link"
      icon={<PiStackBold />}
      onClick={onClick}
      alignSelf="stretch"
      isDisabled={entityCount <= 1 || isBusy}
    />
  );
});

CanvasEntityMergeVisibleButton.displayName = 'CanvasEntityMergeVisibleButton';
