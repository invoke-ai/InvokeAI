import { canvasDownloadedAsImage } from 'features/canvas/store/actions';
import { startAppListening } from '..';
import { log } from 'app/logging/useLogger';
import { downloadBlob } from 'features/canvas/util/downloadBlob';
import { getBaseLayerBlob } from 'features/canvas/util/getBaseLayerBlob';
import { addToast } from 'features/system/store/systemSlice';

const moduleLog = log.child({ namespace: 'canvasSavedToGalleryListener' });

export const addCanvasDownloadedAsImageListener = () => {
  startAppListening({
    actionCreator: canvasDownloadedAsImage,
    effect: async (action, { dispatch, getState }) => {
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

      downloadBlob(blob, 'mergedCanvas.png');
    },
  });
};
