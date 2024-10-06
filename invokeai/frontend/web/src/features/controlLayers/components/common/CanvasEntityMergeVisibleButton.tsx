import { IconButton } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { useAppDispatch } from 'app/store/storeHooks';
import { withResultAsync } from 'common/util/result';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityTypeCount } from 'features/controlLayers/hooks/useEntityTypeCount';
import { inpaintMaskAdded, rasterLayerAdded } from 'features/controlLayers/store/canvasSlice';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiStackBold } from 'react-icons/pi';
import { serializeError } from 'serialize-error';

const log = logger('canvas');

type Props = {
  type: CanvasEntityIdentifier['type'];
};

export const CanvasEntityMergeVisibleButton = memo(({ type }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const canvasManager = useCanvasManager();
  const isBusy = useCanvasIsBusy();
  const entityCount = useEntityTypeCount(type);
  const onClick = useCallback(async () => {
    if (type === 'raster_layer') {
      const rect = canvasManager.stage.getVisibleRect('raster_layer');
      const result = await withResultAsync(() =>
        canvasManager.compositor.rasterizeAndUploadCompositeRasterLayer(rect, { is_intermediate: true })
      );

      if (result.isOk()) {
        dispatch(
          rasterLayerAdded({
            isSelected: true,
            overrides: {
              objects: [imageDTOToImageObject(result.value)],
              position: { x: Math.floor(rect.x), y: Math.floor(rect.y) },
            },
            isMergingVisible: true,
          })
        );
        toast({ title: t('controlLayers.mergeVisibleOk') });
      } else {
        log.error({ error: serializeError(result.error) }, 'Failed to merge visible');
        toast({ title: t('controlLayers.mergeVisibleError'), status: 'error' });
      }
    } else if (type === 'inpaint_mask') {
      const rect = canvasManager.stage.getVisibleRect('inpaint_mask');
      const result = await withResultAsync(() =>
        canvasManager.compositor.rasterizeAndUploadCompositeInpaintMask(rect, false)
      );

      if (result.isOk()) {
        dispatch(
          inpaintMaskAdded({
            isSelected: true,
            overrides: {
              objects: [imageDTOToImageObject(result.value)],
              position: { x: Math.floor(rect.x), y: Math.floor(rect.y) },
            },
            isMergingVisible: true,
          })
        );
        toast({ title: t('controlLayers.mergeVisibleOk') });
      } else {
        log.error({ error: serializeError(result.error) }, 'Failed to merge visible');
        toast({ title: t('controlLayers.mergeVisibleError'), status: 'error' });
      }
    } else {
      log.error({ type }, 'Unsupported type for merge visible');
    }
  }, [canvasManager.compositor, canvasManager.stage, dispatch, t, type]);

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
