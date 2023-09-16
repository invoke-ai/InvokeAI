import { canvasCopiedToClipboard } from 'features/canvas/store/actions';
import { startAppListening } from '..';
import { $logger } from 'app/logging/logger';
import { getBaseLayerBlob } from 'features/canvas/util/getBaseLayerBlob';
import { addToast } from 'features/system/store/systemSlice';
import { copyBlobToClipboard } from 'features/canvas/util/copyBlobToClipboard';
import { t } from 'i18next';

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
            title: t('toast.problemCopyingCanvas'),
            description: t('toast.problemCopyingCanvasDesc'),
            status: 'error',
          })
        );
        return;
      }

      copyBlobToClipboard(blob);

      dispatch(
        addToast({
          title: t('toast.canvasCopiedClipboard'),
          status: 'success',
        })
      );
    },
  });
};
