import { canvasCopiedToClipboard } from 'features/canvas/store/actions';
import { startAppListening } from '..';
import { $logger } from 'app/logging/logger';
import { getBaseLayerBlob } from 'features/canvas/util/getBaseLayerBlob';
import { addToast } from 'features/system/store/systemSlice';
import { copyBlobToClipboard } from 'features/canvas/util/copyBlobToClipboard';

export const addCanvasCopiedToClipboardListener = () => {
  startAppListening({
    actionCreator: canvasCopiedToClipboard,
    effect: async (action, { dispatch, getState }) => {
      const moduleLog = $logger
        .get()
        .child({ namespace: 'canvasCopiedToClipboardListener' });
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

      dispatch(
        addToast({
          title: 'Canvas Copied to Clipboard',
          status: 'success',
        })
      );
    },
  });
};
