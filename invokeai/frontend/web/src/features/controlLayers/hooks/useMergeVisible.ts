import { logger } from 'app/logging/logger';
import { useAppDispatch } from 'app/store/storeHooks';
import { withResultAsync } from 'common/util/result';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import {
  controlLayerAdded,
  inpaintMaskAdded,
  rasterLayerAdded,
  rgAdded,
} from 'features/controlLayers/store/canvasSlice';
import type { CanvasRenderableEntityType } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { useCallback } from 'react';
import { serializeError } from 'serialize-error';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const log = logger('canvas');

export const useMergeVisible = (type: CanvasRenderableEntityType) => {
  const canvasManager = useCanvasManager();
  const dispatch = useAppDispatch();

  const mergeVisible = useCallback(async () => {
    const rect = canvasManager.stage.getVisibleRect(type);
    const adapters = canvasManager.compositor.getVisibleAdaptersOfType(type);
    const result = await withResultAsync(() =>
      canvasManager.compositor.getCompositeImageDTO(
        adapters,
        rect,
        { is_intermediate: true },
        type === 'control_layer' ? { globalCompositeOperation: 'lighter' } : undefined
      )
    );

    if (result.isErr()) {
      log.error({ error: serializeError(result.error) }, 'Failed to merge visible');
      toast({ title: t('controlLayers.mergeVisibleError'), status: 'error' });
      return;
    }

    // All layer types have the same arg - create a new entity with the image as the only object, positioned at the
    // top left corner of the visible rect for the given entity type.
    const arg = {
      isSelected: true,
      overrides: {
        objects: [imageDTOToImageObject(result.value)],
        position: { x: Math.floor(rect.x), y: Math.floor(rect.y) },
      },
    };

    switch (type) {
      case 'raster_layer':
        dispatch(rasterLayerAdded(arg));
        break;
      case 'inpaint_mask':
        dispatch(inpaintMaskAdded(arg));
        break;
      case 'regional_guidance':
        dispatch(rgAdded(arg));
        break;
      case 'control_layer':
        dispatch(controlLayerAdded(arg));
        break;
      default:
        assert<Equals<typeof type, never>>(false, 'Unsupported type for merge visible');
    }

    toast({ title: t('controlLayers.mergeVisibleOk') });
  }, [canvasManager, dispatch, type]);

  return mergeVisible;
};
