import { MenuItem } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { canvasToImageData, getPrefixedId } from 'features/controlLayers/konva/util';
import type { CanvasImageState, Rect } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { PiSelectionBackgroundBold } from 'react-icons/pi';
import { serializeError } from 'serialize-error';

import { toast } from 'features/toast/toast';

const log = logger('canvas');

export const InpaintMaskMenuItemsExtractMaskedArea = memo(() => {
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext('inpaint_mask');
  const isBusy = useCanvasIsBusy();

  const onExtract = useCallback(async () => {
    const maskAdapter = canvasManager.getAdapter(entityIdentifier);
    if (!maskAdapter) {
      log.error({ entityIdentifier }, 'Inpaint mask adapter not found when extracting masked area');
      toast({ status: 'error', title: 'Unable to extract masked area' });
      return;
    }

    try {
      const bbox = canvasManager.stateApi.getBbox();
      const rect: Rect = {
        x: Math.floor(bbox.rect.x),
        y: Math.floor(bbox.rect.y),
        width: Math.floor(bbox.rect.width),
        height: Math.floor(bbox.rect.height),
      };

      if (rect.width <= 0 || rect.height <= 0) {
        toast({ status: 'warning', title: 'Canvas is empty' });
        return;
      }

      const rasterAdapters = canvasManager.compositor.getVisibleAdaptersOfType('raster_layer');

      let compositeImageData: ImageData;
      if (rasterAdapters.length === 0) {
        compositeImageData = new ImageData(rect.width, rect.height);
      } else {
        const compositeCanvas = canvasManager.compositor.getCompositeCanvas(rasterAdapters, rect);
        compositeImageData = canvasToImageData(compositeCanvas);
      }

      const maskCanvas = maskAdapter.getCanvas(rect);
      const maskImageData = canvasToImageData(maskCanvas);

      if (
        maskImageData.width !== compositeImageData.width ||
        maskImageData.height !== compositeImageData.height
      ) {
        log.error(
          {
            maskDimensions: { width: maskImageData.width, height: maskImageData.height },
            compositeDimensions: { width: compositeImageData.width, height: compositeImageData.height },
          },
          'Mask and composite dimensions did not match when extracting masked area'
        );
        toast({ status: 'error', title: 'Unable to extract masked area' });
        return;
      }

      const outputArray = new Uint8ClampedArray(compositeImageData.data.length);
      const compositeArray = compositeImageData.data;
      const maskArray = maskImageData.data;

      for (let i = 0; i < compositeArray.length; i += 4) {
        const maskAlpha = maskArray[i + 3] / 255;

        outputArray[i] = Math.round(compositeArray[i] * maskAlpha);
        outputArray[i + 1] = Math.round(compositeArray[i + 1] * maskAlpha);
        outputArray[i + 2] = Math.round(compositeArray[i + 2] * maskAlpha);
        outputArray[i + 3] = Math.round(compositeArray[i + 3] * maskAlpha);
      }

      const outputImageData = new ImageData(outputArray, rect.width, rect.height);
      const outputCanvas = document.createElement('canvas');
      outputCanvas.width = rect.width;
      outputCanvas.height = rect.height;
      const outputContext = outputCanvas.getContext('2d');

      if (!outputContext) {
        throw new Error('Failed to create canvas context for masked extraction');
      }

      outputContext.putImageData(outputImageData, 0, 0);

      const imageState: CanvasImageState = {
        id: getPrefixedId('image'),
        type: 'image',
        image: {
          dataURL: outputCanvas.toDataURL('image/png'),
          width: rect.width,
          height: rect.height,
        },
      };

      const addAfter = canvasManager.stateApi.getRasterLayersState().entities.at(-1)?.id;

      canvasManager.stateApi.addRasterLayer({
        overrides: {
          objects: [imageState],
          position: { x: rect.x, y: rect.y },
        },
        isSelected: true,
        addAfter,
      });
    } catch (error) {
      log.error({ error: serializeError(error as Error) }, 'Failed to extract masked area to raster layer');
      toast({ status: 'error', title: 'Unable to extract masked area' });
    }
  }, [canvasManager, entityIdentifier]);

  return (
    <MenuItem
      onClick={onExtract}
      icon={<PiSelectionBackgroundBold />}
      isDisabled={isBusy}
    >
      Extract masked area to new layer
    </MenuItem>
  );
});

InpaintMaskMenuItemsExtractMaskedArea.displayName = 'InpaintMaskMenuItemsExtractMaskedArea';

