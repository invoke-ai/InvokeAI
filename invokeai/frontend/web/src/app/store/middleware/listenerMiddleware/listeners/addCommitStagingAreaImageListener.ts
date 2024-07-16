import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import {
  layerAdded,
  layerImageAdded,
  sessionStagingAreaImageAccepted,
  sessionStagingAreaReset,
} from 'features/controlLayers/store/canvasV2Slice';
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
    effect: async (action, api) => {
      const { index } = action.payload;
      const { layers, selectedEntityIdentifier } = api.getState().canvasV2;
      let layer = layers.entities.find((layer) => layer.id === selectedEntityIdentifier?.id);

      if (!layer) {
        layer = layers.entities[0];
      }

      if (!layer) {
        // We need to create a new layer to add the accepted image
        api.dispatch(layerAdded());
        layer = api.getState().canvasV2.layers.entities[0];
      }

      const stagedImage = api.getState().canvasV2.session.stagedImages[index];

      assert(stagedImage, 'No staged image found to accept');
      assert(layer, 'No layer found to stage image');

      const { id } = layer;

      api.dispatch(
        layerImageAdded({
          id,
          imageDTO: stagedImage.imageDTO,
          pos: { x: stagedImage.rect.x - layer.x, y: stagedImage.rect.y - layer.y },
        })
      );
      api.dispatch(sessionStagingAreaReset());
    },
  });
};
