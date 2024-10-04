import { useAppSelector } from 'app/store/storeHooks';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterInpaintMask } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterInpaintMask';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasEntityAdapterRegionalGuidance } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRegionalGuidance';
import { canvasToBlob } from 'features/controlLayers/konva/util';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useUploadImageMutation } from 'services/api/endpoints/images';

export const useSaveLayerToAssets = () => {
  const { t } = useTranslation();
  const [uploadImage] = useUploadImageMutation();
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);

  const saveLayerToAssets = useCallback(
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
        const file = new File([blob], `layer-${adapter.id}.png`, { type: 'image/png' });
        await uploadImage({
          file,
          image_category: 'user',
          is_intermediate: false,
          postUploadAction: { type: 'TOAST' },
          board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
        });

        toast({
          status: 'info',
          title: t('toast.layerSavedToAssets'),
        });
      } catch (error) {
        toast({
          status: 'error',
          title: t('toast.problemSavingLayer'),
        });
      }
    },
    [t, autoAddBoardId, uploadImage]
  );

  return saveLayerToAssets;
};
