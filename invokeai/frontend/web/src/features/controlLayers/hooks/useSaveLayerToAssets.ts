import { useAppSelector } from 'app/store/storeHooks';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterInpaintMask } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterInpaintMask';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasEntityAdapterRegionalGuidance } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRegionalGuidance';
import { canvasToBlob } from 'features/controlLayers/konva/util';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { useCallback } from 'react';
import { uploadImage } from 'services/api/endpoints/images';

export const useSaveLayerToAssets = () => {
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
      const canvas = adapter.getCanvas();
      const blob = await canvasToBlob(canvas);
      const file = new File([blob], `layer-${adapter.id}.png`, { type: 'image/png' });
      uploadImage({
        file,
        image_category: 'user',
        is_intermediate: false,
        board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
      });
    },
    [autoAddBoardId]
  );

  return saveLayerToAssets;
};
