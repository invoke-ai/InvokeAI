import { logger } from 'app/logging/logger';
import { withResultAsync } from 'common/util/result';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterInpaintMask } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterInpaintMask';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasEntityAdapterRegionalGuidance } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRegionalGuidance';
import { canvasToBlob } from 'features/controlLayers/konva/util';
import { copyBlobToClipboard } from 'features/system/util/copyBlobToClipboard';
import { toast } from 'features/toast/toast';
import { startCase } from 'lodash-es';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { serializeError } from 'serialize-error';

const log = logger('canvas');

export const useCopyLayerToClipboard = () => {
  const { t } = useTranslation();
  const copyLayerToCipboard = useCallback(
    async (
      adapter:
        | CanvasEntityAdapterRasterLayer
        | CanvasEntityAdapterControlLayer
        | CanvasEntityAdapterInpaintMask
        | CanvasEntityAdapterRegionalGuidance
        | null
    ) => {
      if (!adapter) {
        return;
      }

      const result = await withResultAsync(async () => {
        const canvas = adapter.getCanvas();
        const blob = await canvasToBlob(canvas);
        copyBlobToClipboard(blob);
      });

      if (result.isOk()) {
        log.trace('Layer copied to clipboard');
        toast({
          status: 'info',
          title: t('toast.layerCopiedToClipboard'),
        });
      } else {
        log.error({ error: serializeError(result.error) }, 'Problem copying layer to clipboard');
        toast({
          status: 'error',
          title: t('toast.problemCopyingLayer'),
        });
      }
    },
    [t]
  );

  return copyLayerToCipboard;
};

export const useCopyCanvasToClipboard = (region: 'canvas' | 'bbox') => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const copyCanvasToClipboard = useCallback(async () => {
    const rect =
      region === 'bbox'
        ? canvasManager.stateApi.getBbox().rect
        : canvasManager.compositor.getVisibleRectOfType('raster_layer');

    if (rect.width === 0 || rect.height === 0) {
      toast({
        title: t('controlLayers.copyRegionError', { region: startCase(region) }),
        description: t('controlLayers.regionIsEmpty'),
        status: 'warning',
      });
      return;
    }

    const result = await withResultAsync(async () => {
      const rasterAdapters = canvasManager.compositor.getVisibleAdaptersOfType('raster_layer');
      const canvasElement = canvasManager.compositor.getCompositeCanvas(rasterAdapters, rect);
      const blob = await canvasToBlob(canvasElement);
      copyBlobToClipboard(blob);
    });

    if (result.isOk()) {
      toast({ title: t('controlLayers.regionCopiedToClipboard', { region: startCase(region) }) });
    } else {
      log.error({ error: serializeError(result.error) }, 'Failed to save canvas to gallery');
      toast({ title: t('controlLayers.copyRegionError', { region: startCase(region) }), status: 'error' });
    }
  }, [canvasManager.compositor, canvasManager.stateApi, region, t]);

  return copyCanvasToClipboard;
};
