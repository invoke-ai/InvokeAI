import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import {
  $lastProgressEvent,
  layerAdded,
  sessionStagingAreaImageAccepted,
  sessionStagingAreaReset,
} from 'features/controlLayers/store/canvasV2Slice';
import type { CanvasLayerState } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/types';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { queueApi } from 'services/api/endpoints/queue';
import { assert } from 'tsafe';

export const addStagingListeners = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: sessionStagingAreaReset,
    effect: async (_, { dispatch }) => {
      const log = logger('canvas');

      try {
        const req = dispatch(
          queueApi.endpoints.cancelByBatchOrigin.initiate(
            { origin: 'canvas' },
            { fixedCacheKey: 'cancelByBatchOrigin' }
          )
        );
        const { canceled } = await req.unwrap();
        req.reset();

        $lastProgressEvent.set(null);

        if (canceled > 0) {
          log.debug(`Canceled ${canceled} canvas batches`);
          toast({
            id: 'CANCEL_BATCH_SUCCEEDED',
            title: t('queue.cancelBatchSucceeded'),
            status: 'success',
          });
        }
      } catch {
        log.error('Failed to cancel canvas batches');
        toast({
          id: 'CANCEL_BATCH_FAILED',
          title: t('queue.cancelBatchFailed'),
          status: 'error',
        });
      }
    },
  });

  startAppListening({
    actionCreator: sessionStagingAreaImageAccepted,
    effect: (action, api) => {
      const { index } = action.payload;
      const state = api.getState();
      const stagingAreaImage = state.canvasV2.session.stagedImages[index];

      assert(stagingAreaImage, 'No staged image found to accept');
      const { x, y } = state.canvasV2.bbox.rect;

      const { imageDTO, offsetX, offsetY } = stagingAreaImage;
      const imageObject = imageDTOToImageObject(imageDTO);
      const overrides: Partial<CanvasLayerState> = {
        position: { x: x + offsetX, y: y + offsetY },
        objects: [imageObject],
      };

      api.dispatch(layerAdded({ overrides }));
      api.dispatch(sessionStagingAreaReset());
    },
  });
};
