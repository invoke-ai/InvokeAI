import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { canvasImageToControlAdapter } from 'features/canvas/store/actions';
import { getBaseLayerBlob } from 'features/canvas/util/getBaseLayerBlob';
import { caImageChanged } from 'features/controlLayers/store/canvasV2Slice';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { imagesApi } from 'services/api/endpoints/images';

export const addCanvasImageToControlNetListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: canvasImageToControlAdapter,
    effect: async (action, { dispatch, getState }) => {
      const log = logger('canvas');
      const state = getState();
      const { id } = action.payload;

      let blob: Blob;
      try {
        blob = await getBaseLayerBlob(state, true);
      } catch (err) {
        log.error(String(err));
        toast({
          id: 'PROBLEM_SAVING_CANVAS',
          title: t('toast.problemSavingCanvas'),
          description: t('toast.problemSavingCanvasDesc'),
          status: 'error',
        });
        return;
      }

      const { autoAddBoardId } = state.gallery;

      const imageDTO = await dispatch(
        imagesApi.endpoints.uploadImage.initiate({
          file: new File([blob], 'savedCanvas.png', {
            type: 'image/png',
          }),
          image_category: 'control',
          is_intermediate: true,
          board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
          crop_visible: false,
          postUploadAction: {
            type: 'TOAST',
            title: t('toast.canvasSentControlnetAssets'),
          },
        })
      ).unwrap();

      dispatch(caImageChanged({ id, imageDTO }));
    },
  });
};
