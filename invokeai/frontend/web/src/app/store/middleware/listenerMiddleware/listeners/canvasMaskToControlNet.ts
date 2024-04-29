import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { canvasMaskToControlAdapter } from 'features/canvas/store/actions';
import { getCanvasData } from 'features/canvas/util/getCanvasData';
import { controlAdapterImageChanged } from 'features/controlAdapters/store/controlAdaptersSlice';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';
import { imagesApi } from 'services/api/endpoints/images';

export const addCanvasMaskToControlNetListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: canvasMaskToControlAdapter,
    effect: async (action, { dispatch, getState }) => {
      const log = logger('canvas');
      const state = getState();
      const { id } = action.payload;
      const canvasBlobsAndImageData = await getCanvasData(
        state.canvas.layerState,
        state.canvas.boundingBoxCoordinates,
        state.canvas.boundingBoxDimensions,
        state.canvas.isMaskEnabled,
        state.canvas.shouldPreserveMaskedArea
      );

      if (!canvasBlobsAndImageData) {
        return;
      }

      const { maskBlob } = canvasBlobsAndImageData;

      if (!maskBlob) {
        log.error('Problem getting mask layer blob');
        dispatch(
          addToast({
            title: t('toast.problemImportingMask'),
            description: t('toast.problemImportingMaskDesc'),
            status: 'error',
          })
        );
        return;
      }

      const { autoAddBoardId } = state.gallery;

      const imageDTO = await dispatch(
        imagesApi.endpoints.uploadImage.initiate({
          file: new File([maskBlob], 'canvasMaskImage.png', {
            type: 'image/png',
          }),
          image_category: 'mask',
          is_intermediate: true,
          board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
          crop_visible: false,
          postUploadAction: {
            type: 'TOAST',
            toastOptions: { title: t('toast.maskSentControlnetAssets') },
          },
        })
      ).unwrap();

      dispatch(
        controlAdapterImageChanged({
          id,
          controlImage: imageDTO,
        })
      );
    },
  });
};
