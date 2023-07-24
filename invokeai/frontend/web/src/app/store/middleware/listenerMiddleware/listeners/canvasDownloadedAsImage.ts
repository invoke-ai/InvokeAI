import { canvasDownloadedAsImage } from 'features/canvas/store/actions';
import { startAppListening } from '..';
import { $logger } from 'app/logging/logger';
import { downloadBlob } from 'features/canvas/util/downloadBlob';
import { getBaseLayerBlob } from 'features/canvas/util/getBaseLayerBlob';
import { addToast } from 'features/system/store/systemSlice';

export const addCanvasDownloadedAsImageListener = () => {
  startAppListening({
    actionCreator: canvasDownloadedAsImage,
    effect: async (action, { dispatch, getState }) => {
      const moduleLog = $logger
        .get()
        .child({ namespace: 'canvasSavedToGalleryListener' });
      const state = getState();

      const blob = await getBaseLayerBlob(state);

      if (!blob) {
        moduleLog.error('Problem getting base layer blob');
        dispatch(
          addToast({
            title: 'Problem Downloading Canvas',
            description: 'Unable to export base layer',
            status: 'error',
          })
        );
        return;
      }

      downloadBlob(blob, 'canvas.png');
      dispatch(addToast({ title: 'Canvas Downloaded', status: 'success' }));
    },
  });
};
