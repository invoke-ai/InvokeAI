import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { canvasToBlob, canvasToImageData } from 'features/controlLayers/konva/util';
import { entityRasterized } from 'features/controlLayers/store/canvasSlice';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { uploadImage } from 'services/api/endpoints/images';

export const useInvertMask = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const canvasManager = useCanvasManager();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);

  const invertMask = useCallback(async () => {
    try {

      const adapters = canvasManager.compositor.getVisibleAdaptersOfType('inpaint_mask');

      if (adapters.length === 0) {
        toast({
          id: 'NO_VISIBLE_MASKS',
          title: t('toast.noVisibleMasks'),
          description: t('toast.noVisibleMasksDesc'),
          status: 'warning',
        });
        return;
      }

      const fullCanvas = document.createElement('canvas');
      fullCanvas.width = bboxRect.width;
      fullCanvas.height = bboxRect.height;
      const fullCtx = fullCanvas.getContext('2d');

      if (!fullCtx) {
        throw new Error('Failed to get canvas context');
      }

      // Fill with transparent black (no mask)
      fullCtx.fillStyle = 'rgba(0, 0, 0, 0)';
      fullCtx.fillRect(0, 0, bboxRect.width, bboxRect.height);

      // Get the visible masks rect
      const visibleMasksRect = canvasManager.compositor.getVisibleRectOfType('inpaint_mask');

      // Only composite if there's a visible rect
      if (visibleMasksRect.width > 0 && visibleMasksRect.height > 0) {
        // Get composite of masks in their original position
        const compositeCanvas = canvasManager.compositor.getCompositeCanvas(adapters, visibleMasksRect);

        // Draw the composite onto the full canvas at the correct position
        const offsetX = visibleMasksRect.x - bboxRect.x;
        const offsetY = visibleMasksRect.y - bboxRect.y;
        fullCtx.drawImage(compositeCanvas, offsetX, offsetY);
      }

      // Get image data and invert
      const imageData = canvasToImageData(fullCanvas);
      const data = imageData.data;

      // Invert alpha values (where current masks are opaque, inverted mask will be transparent)
      for (let i = 3; i < data.length; i += 4) {
        data[i] = 255 - data[i]; // Invert alpha
      }

      // Put the inverted data back
      fullCtx.putImageData(imageData, 0, 0);

      // Convert to blob and upload
      const blob = await canvasToBlob(fullCanvas);
      const imageDTO = await uploadImage({
        file: new File([blob], 'inverted-mask.png', { type: 'image/png' }),
        image_category: 'general',
        is_intermediate: true,
        silent: true,
      });

      // Create image object from the inverted mask
      const imageObject = imageDTOToImageObject(imageDTO);

      // Replace the selected mask's objects with the inverted mask
      if (selectedEntityIdentifier) {
        dispatch(
          entityRasterized({
            entityIdentifier: selectedEntityIdentifier,
            imageObject,
            position: { x: bboxRect.x, y: bboxRect.y },
            replaceObjects: true,
            isSelected: true,
          })
        );
      }

      toast({
        id: 'MASK_INVERTED',
        title: t('toast.maskInverted'),
        status: 'success',
      });
    } catch (error) {
      toast({
        id: 'MASK_INVERT_FAILED',
        title: t('toast.maskInvertFailed'),
        description: String(error),
        status: 'error',
      });
    }
  }, [canvasManager, dispatch, selectedEntityIdentifier, t]);

  return invertMask;
};
