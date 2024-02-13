import { $logger } from 'app/logging/logger';
import { canvasCopiedToClipboard } from 'features/canvas/store/actions';
import { getBaseLayerBlob } from 'features/canvas/util/getBaseLayerBlob';
import { addToast } from 'features/system/store/systemSlice';
import { copyBlobToClipboard } from 'features/system/util/copyBlobToClipboard';
import { t } from 'i18next';

import { startAppListening } from '..';

export const addCanvasCopiedToClipboardListener = () => {
  startAppListening({
    actionCreator: canvasCopiedToClipboard,
    effect: async (action, { dispatch, getState }) => {
      const moduleLog = $logger.get().child({ namespace: 'canvasCopiedToClipboardListener' });
      const state = getState();

      try {
        const blob = getBaseLayerBlob(state);

        copyBlobToClipboard(blob);
      } catch (err) {
        moduleLog.error(String(err));
        dispatch(
          addToast({
            title: t('toast.problemCopyingCanvas'),
            description: t('toast.problemCopyingCanvasDesc'),
            status: 'error',
          })
        );
        return;
      }

      dispatch(
        addToast({
          title: t('toast.canvasCopiedClipboard'),
          status: 'success',
        })
      );
    },
  });
};
