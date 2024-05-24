import { $logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { canvasCopiedToClipboard } from 'features/canvas/store/actions';
import { getBaseLayerBlob } from 'features/canvas/util/getBaseLayerBlob';
import { copyBlobToClipboard } from 'features/system/util/copyBlobToClipboard';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';

export const addCanvasCopiedToClipboardListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: canvasCopiedToClipboard,
    effect: async (action, { getState }) => {
      const moduleLog = $logger.get().child({ namespace: 'canvasCopiedToClipboardListener' });
      const state = getState();

      try {
        const blob = getBaseLayerBlob(state);

        copyBlobToClipboard(blob);
      } catch (err) {
        moduleLog.error(String(err));
        toast({
          id: 'CANVAS_COPY_FAILED',
          title: t('toast.problemCopyingCanvas'),
          description: t('toast.problemCopyingCanvasDesc'),
          status: 'error',
        });
        return;
      }

      toast({
        id: 'CANVAS_COPY_SUCCEEDED',
        title: t('toast.canvasCopiedClipboard'),
        status: 'success',
      });
    },
  });
};
