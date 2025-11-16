import { MenuItem } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { canvasToImageData, getPrefixedId } from 'features/controlLayers/konva/util';
import type { CanvasImageState, Rect } from 'features/controlLayers/store/types';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { PiSelectionBackgroundBold } from 'react-icons/pi';
import { serializeError } from 'serialize-error';

const log = logger('canvas');

export const InpaintMaskMenuItemsExtractMaskedArea = memo(() => {
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext('inpaint_mask');
  const isBusy = useCanvasIsBusy();

  const onExtract = useCallback(() => {
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

      // Ensure both composite and mask image data exist and agree on dimensions.
      if (
        !compositeImageData ||
        !maskImageData ||
        maskImageData.width !== compositeImageData.width ||
        maskImageData.height !== compositeImageData.height
      ) {
        log.error(
          {
            hasComposite: !!compositeImageData,
            hasMask: !!maskImageData,
            maskDimensions: maskImageData ? { width: maskImageData.width, height: maskImageData.height } : null,
            compositeDimensions: compositeImageData
              ? { width: compositeImageData.width, height: compositeImageData.height }
              : null,
          },
          'Mask and composite dimensions did not match or image data missing when extracting masked area'
        );
        toast({ status: 'error', title: 'Unable to extract masked area' });
        return;
      }

      // At this point both image buffers are guaranteed to be valid and dimensionally aligned.
      const compositeArray = compositeImageData.data;
      const maskArray = maskImageData.data;

      // Prepare output pixel buffer.
      const outputArray = new Uint8ClampedArray(compositeArray.length);

      // Apply the mask alpha only to the alpha channel.
      // Do NOT multiply RGB by maskAlpha to avoid dark fringe artifacts around mask edges.
      for (let i = 0; i < compositeArray.length; i += 4) {
        // Read original composite pixel, defaulting to 0 to satisfy strict indexed access rules.
        const r = compositeArray[i] ?? 0;
        const g = compositeArray[i + 1] ?? 0;
        const b = compositeArray[i + 2] ?? 0;
        const a = compositeArray[i + 3] ?? 0;

        // Extract mask alpha (0..255 → 0..1).
        const maskAlpha = (maskArray[i + 3] ?? 0) / 255;

        // Preserve original RGB values.
        outputArray[i] = r;
        outputArray[i + 1] = g;
        outputArray[i + 2] = b;

        // Mask only the alpha channel to avoid halo artifacts.
        outputArray[i + 3] = Math.round(a * maskAlpha);
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

      // Insert the new raster layer at the top of the raster layer stack.
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
    <MenuItem onClick={onExtract} icon={<PiSelectionBackgroundBold />} isDisabled={isBusy}>
      Extract masked area to new layer
    </MenuItem>
  );
});

InpaintMaskMenuItemsExtractMaskedArea.displayName = 'InpaintMaskMenuItemsExtractMaskedArea';
