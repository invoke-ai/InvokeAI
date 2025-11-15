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
    // The active inpaint mask layer is required to build the mask used for extraction.
    const maskAdapter = canvasManager.getAdapter(entityIdentifier);
    if (!maskAdapter) {
      log.error({ entityIdentifier }, 'Inpaint mask adapter not found when extracting masked area');
      toast({ status: 'error', title: 'Unable to extract masked area' });
      return;
    }

    try {
      // Get the canvas bounding box so the raster extraction respects the visible canvas bounds.
      const bbox = canvasManager.stateApi.getBbox();
      const rect: Rect = {
        x: Math.floor(bbox.rect.x),
        y: Math.floor(bbox.rect.y),
        width: Math.floor(bbox.rect.width),
        height: Math.floor(bbox.rect.height),
      };

      // Abort when the canvas is effectively empty—no pixels to extract.
      if (rect.width <= 0 || rect.height <= 0) {
        toast({ status: 'warning', title: 'Canvas is empty' });
        return;
      }

      // Gather the visible raster layer adapters so we can composite them into a single bitmap.
      const rasterAdapters = canvasManager.compositor.getVisibleAdaptersOfType('raster_layer');

      let compositeImageData: ImageData;
      if (rasterAdapters.length === 0) {
        // No visible raster layers—create a transparent buffer that matches the canvas bounds.
        compositeImageData = new ImageData(rect.width, rect.height);
      } else {
        // Render the visible raster layers into an offscreen canvas restricted to the canvas bounds.
        const compositeCanvas = canvasManager.compositor.getCompositeCanvas(rasterAdapters, rect);
        compositeImageData = canvasToImageData(compositeCanvas);
      }

      // Render the inpaint mask layer into a canvas so we have the alpha data that defines the mask.
      const maskCanvas = maskAdapter.getCanvas(rect);
      const maskImageData = canvasToImageData(maskCanvas);

      if (
        maskImageData.width !== compositeImageData.width ||
        maskImageData.height !== compositeImageData.height
      ) {
        // Bail out if the mask and composite buffers disagree on dimensions.
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

      if (!compositeArray || !maskArray) {
        toast({ status: 'error', title: 'Cannot extract: image or mask data is missing.' });
        return;
      }

      // Apply the mask alpha channel to each pixel in the composite, keeping RGB but zeroing alpha outside the mask.
      for (let i = 0; i < compositeArray.length; i += 4) {
        const maskAlpha = (maskArray[i + 3] ?? 0) / 255;
        outputArray[i] = Math.round((compositeArray[i] ?? 0) * maskAlpha);
        outputArray[i + 1] = Math.round((compositeArray[i + 1] ?? 0) * maskAlpha);
        outputArray[i + 2] = Math.round((compositeArray[i + 2] ?? 0) * maskAlpha);
        outputArray[i + 3] = Math.round((compositeArray[i + 3] ?? 0) * maskAlpha);
      }

      // Package the masked pixels into an ImageData and draw them to an offscreen canvas.
      const outputImageData = new ImageData(outputArray, rect.width, rect.height);
      const outputCanvas = document.createElement('canvas');
      outputCanvas.width = rect.width;
      outputCanvas.height = rect.height;
      const outputContext = outputCanvas.getContext('2d');

      if (!outputContext) {
        throw new Error('Failed to create canvas context for masked extraction');
      }

      outputContext.putImageData(outputImageData, 0, 0);

      // Convert the offscreen canvas into an Invoke canvas image state for insertion into the layer stack.
      const imageState: CanvasImageState = {
        id: getPrefixedId('image'),
        type: 'image',
        image: {
          dataURL: outputCanvas.toDataURL('image/png'),
          width: rect.width,
          height: rect.height,
        },
      };

      // Insert the new raster layer just after the last existing raster layer so it appears above the mask.
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

