import { $logger } from 'app/logging/logger';
import { canvasDownloadedAsImage } from 'features/canvas/store/actions';
import { downloadBlob } from 'features/canvas/util/downloadBlob';
import { getBaseLayerBlob } from 'features/canvas/util/getBaseLayerBlob';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';

import { startAppListening } from '..';

export const addCanvasDownloadedAsImageListener = () => {
  startAppListening({
    actionCreator: canvasDownloadedAsImage,
    effect: async (action, { dispatch, getState }) => {
      const moduleLog = $logger.get().child({ namespace: 'canvasSavedToGalleryListener' });
      const state = getState();

      let blob;
      try {
        blob = await getBaseLayerBlob(state);
      } catch (err) {
        moduleLog.error(String(err));
        dispatch(
          addToast({
            title: t('toast.problemDownloadingCanvas'),
            description: t('toast.problemDownloadingCanvasDesc'),
            status: 'error',
          })
        );
        return;
      }

      downloadBlob(blob, 'canvas.png');
      dispatch(addToast({ title: t('toast.canvasDownloaded'), status: 'success' }));
    },
  });
};
