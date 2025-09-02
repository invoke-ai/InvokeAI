import { useCanvasContext } from 'features/controlLayers/contexts/CanvasInstanceContext';
import { canvasToBlob, canvasToImageData } from 'features/controlLayers/konva/util';
import { instanceActions } from 'features/controlLayers/store/canvasInstanceSlice';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { uploadImage } from 'services/api/endpoints/images';

export const useInvertMask = () => {
  const { t } = useTranslation();
  const { dispatch, manager: canvasManager, useSelector } = useCanvasContext();
  const selectedEntityIdentifier = useSelector((state) => state.selectedEntityIdentifier);

  const invertMask = useCallback(async () => {
    try {
      const bbox = canvasManager.stateApi.getBbox();
      if (!bbox) {
        toast({
          id: 'NO_BBOX',
          title: t('toast.noBbox'),
          description: t('toast.noBboxDesc'),
          status: 'warning',
        });
        return;
      }
      const bboxRect = bbox.rect;

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

      fullCtx.fillStyle = 'rgba(0, 0, 0, 0)';
      fullCtx.fillRect(0, 0, bboxRect.width, bboxRect.height);

      const visibleMasksRect = canvasManager.compositor.getVisibleRectOfType('inpaint_mask');

      if (visibleMasksRect.width > 0 && visibleMasksRect.height > 0) {
        const compositeCanvas = canvasManager.compositor.getCompositeCanvas(adapters, visibleMasksRect);

        const offsetX = visibleMasksRect.x - bboxRect.x;
        const offsetY = visibleMasksRect.y - bboxRect.y;
        fullCtx.drawImage(compositeCanvas, offsetX, offsetY);
      }

      const imageData = canvasToImageData(fullCanvas);
      const data = imageData.data;

      for (let i = 3; i < data.length; i += 4) {
        data[i] = 255 - (data[i] ?? 0); // Invert alpha
      }

      fullCtx.putImageData(imageData, 0, 0);

      const blob = await canvasToBlob(fullCanvas);
      const imageDTO = await uploadImage({
        file: new File([blob], 'inverted-mask.png', { type: 'image/png' }),
        image_category: 'general',
        is_intermediate: true,
        silent: true,
      });

      const imageObject = imageDTOToImageObject(imageDTO);

      if (selectedEntityIdentifier) {
        dispatch(
          instanceActions.entityRasterized({
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
