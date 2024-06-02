import { $logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { canvasDownloadedAsImage } from 'features/canvas/store/actions';
import { downloadBlob } from 'features/canvas/util/downloadBlob';
import { getBaseLayerBlob } from 'features/canvas/util/getBaseLayerBlob';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';

export const addCanvasDownloadedAsImageListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: canvasDownloadedAsImage,
    effect: async (action, { getState }) => {
      const moduleLog = $logger.get().child({ namespace: 'canvasSavedToGalleryListener' });
      const state = getState();

      let blob;
      try {
        blob = await getBaseLayerBlob(state);
      } catch (err) {
        moduleLog.error(String(err));
        toast({
          id: 'CANVAS_DOWNLOAD_FAILED',
          title: t('toast.problemDownloadingCanvas'),
          description: t('toast.problemDownloadingCanvasDesc'),
          status: 'error',
        });
        return;
      }

      downloadBlob(blob, 'canvas.png');
      toast({ id: 'CANVAS_DOWNLOAD_SUCCEEDED', title: t('toast.canvasDownloaded'), status: 'success' });
    },
  });
};
