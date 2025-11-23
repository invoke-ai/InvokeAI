import { MenuItem } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { canvasToBlob, canvasToImageData } from 'features/controlLayers/konva/util';
import type { CanvasImageState, Rect } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSelectionBackgroundBold } from 'react-icons/pi';
import { serializeError } from 'serialize-error';
import { uploadImage } from 'services/api/endpoints/images';

const log = logger('canvas');

export const InpaintMaskMenuItemsExtractMaskedArea = memo(() => {
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext('inpaint_mask');
  const isBusy = useCanvasIsBusy();
  const { t } = useTranslation();

  const onExtract = useCallback(() => {
    void (async () => {
      // The active inpaint mask layer is required to build the mask used for extraction.
      const maskAdapter = canvasManager.getAdapter(entityIdentifier);
      if (!maskAdapter) {
        log.error({ entityIdentifier }, 'Inpaint mask adapter not found when extracting masked area');
        toast({ status: 'error', title: 'Unable to extract masked area.' });
        return;
      }

      try {
        // Use the mask's bounding box in stage coordinates to constrain the extraction region.
        const maskPixelRect = maskAdapter.transformer.$pixelRect.get();
        const maskPosition = maskAdapter.state.position;
        const rect: Rect = {
          x: Math.floor(maskPosition.x + maskPixelRect.x),
          y: Math.floor(maskPosition.y + maskPixelRect.y),
          width: Math.floor(maskPixelRect.width),
          height: Math.floor(maskPixelRect.height),
        };

        // Abort when the canvas is effectively empty—no pixels to extract.
        if (rect.width <= 0 || rect.height <= 0) {
          toast({ status: 'warning', title: 'Canvas is empty.' });
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

        if (maskImageData.width !== compositeImageData.width || maskImageData.height !== compositeImageData.height) {
          // Bail out if the mask and composite buffers disagree on dimensions.
          log.error(
            {
              maskDimensions: { width: maskImageData.width, height: maskImageData.height },
              compositeDimensions: { width: compositeImageData.width, height: compositeImageData.height },
            },
            'Mask and composite dimensions did not match when extracting masked area'
          );
          toast({ status: 'error', title: 'Unable to extract masked area.' });
          return;
        }

        const compositeArray = compositeImageData.data;
        const maskArray = maskImageData.data;

        if (!compositeArray || !maskArray) {
          toast({ status: 'error', title: 'Cannot extract: image or mask data is missing.' });
          return;
        }

        const outputArray = new Uint8ClampedArray(compositeArray.length);

        // Apply the mask alpha channel to each pixel in the composite, keeping RGB untouched and only masking alpha.
        for (let i = 0; i < compositeArray.length; i += 4) {
          const maskAlpha = (maskArray[i + 3] ?? 0) / 255 || 0;
          outputArray[i] = compositeArray[i] ?? 0;
          outputArray[i + 1] = compositeArray[i + 1] ?? 0;
          outputArray[i + 2] = compositeArray[i + 2] ?? 0;
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

        // Upload the extracted canvas region as a real image resource and returns image_name

        const blob = await canvasToBlob(outputCanvas);

        const imageDTO = await uploadImage({
          file: new File([blob], 'inpaint-extract.png', { type: 'image/png' }),
          image_category: 'general',
          is_intermediate: true,
          silent: true,
        });

        // Convert the uploaded image DTO into the canvas image state to avoid serializing the PNG in client state.
        const imageState: CanvasImageState = imageDTOToImageObject(imageDTO);

        // Insert the new raster layer so it appears at the top of raster layer goup.
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
        toast({ status: 'error', title: 'Unable to extract masked area.' });
      }
    })();
  }, [canvasManager, entityIdentifier]);

  return (
    <MenuItem onClick={onExtract} icon={<PiSelectionBackgroundBold />} isDisabled={isBusy}>
      {t('controlLayers.extractRegion')}
    </MenuItem>
  );
});

InpaintMaskMenuItemsExtractMaskedArea.displayName = 'InpaintMaskMenuItemsExtractMaskedArea';
