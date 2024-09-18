import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterInpaintMask } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterInpaintMask';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasEntityAdapterRegionalGuidance } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRegionalGuidance';
import { canvasToBlob } from 'features/controlLayers/konva/util';
import { copyBlobToClipboard } from 'features/system/util/copyBlobToClipboard';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

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
      try {
        const canvas = adapter.getCanvas();
        const blob = await canvasToBlob(canvas);
        copyBlobToClipboard(blob);
        toast({
          status: 'info',
          title: t('toast.layerCopiedToClipboard'),
        });
      } catch (error) {
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
