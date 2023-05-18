import { canvasCopiedToClipboard } from 'features/canvas/store/actions';
import { startAppListening } from '..';
import { log } from 'app/logging/useLogger';
import { getBaseLayerBlob } from 'features/canvas/util/getBaseLayerBlob';
import { addToast } from 'features/system/store/systemSlice';
import { copyBlobToClipboard } from 'features/canvas/util/copyBlobToClipboard';

const moduleLog = log.child({ namespace: 'canvasCopiedToClipboardListener' });

export const addCanvasCopiedToClipboardListener = () => {
  startAppListening({
    actionCreator: canvasCopiedToClipboard,
    effect: async (action, { dispatch, getState }) => {
      const state = getState();

      const blob = await getBaseLayerBlob(state);

      if (!blob) {
        moduleLog.error('Problem getting base layer blob');
        dispatch(
          addToast({
            title: 'Problem Copying Canvas',
            description: 'Unable to export base layer',
            status: 'error',
          })
        );
        return;
      }

      copyBlobToClipboard(blob);
    },
  });
};
