import { logger } from 'app/logging/logger';
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
import { serializeError } from 'serialize-error';
import { uploadImage } from 'services/api/endpoints/images';

const log = logger('canvas');

export const useSaveLayerToAssets = () => {
  const { t } = useTranslation();
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
        uploadImage({
          file,
          image_category: 'user',
          is_intermediate: false,
          board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
        });
      } catch (error) {
        log.error({ error: serializeError(error) }, 'Problem copying layer to clipboard');
        toast({
          status: 'error',
          title: t('toast.problemSavingLayer'),
        });
      }
    },
    [autoAddBoardId, t]
  );

  return saveLayerToAssets;
};
