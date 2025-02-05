import { logger } from 'app/logging/logger';
import { withResultAsync } from 'common/util/result';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterInpaintMask } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterInpaintMask';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasEntityAdapterRegionalGuidance } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRegionalGuidance';
import { canvasToBlob } from 'features/controlLayers/konva/util';
import { copyBlobToClipboard } from 'features/system/util/copyBlobToClipboard';
import { toast } from 'features/toast/toast';
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
