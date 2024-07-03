import { isAnyOf } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import {
  layerAdded,
  layerImageAdded,
  stagingAreaCanceledStaging,
  stagingAreaImageAccepted,
} from 'features/controlLayers/store/canvasV2Slice';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { queueApi } from 'services/api/endpoints/queue';
import { assert } from 'tsafe';

export const addStagingListeners = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: isAnyOf(stagingAreaCanceledStaging, stagingAreaImageAccepted),
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
    actionCreator: stagingAreaImageAccepted,
    effect: async (action, api) => {
      const { imageDTO } = action.payload;
      const { layers, selectedEntityIdentifier, bbox } = api.getState().canvasV2;
      let layer = layers.entities.find((layer) => layer.id === selectedEntityIdentifier?.id);

      if (!layer) {
        layer = layers.entities[0];
      }

      if (!layer) {
        // We need to create a new layer to add the accepted image
        api.dispatch(layerAdded());
        layer = api.getState().canvasV2.layers.entities[0];
      }

      assert(layer, 'No layer found to stage image');

      const { id } = layer;

      api.dispatch(layerImageAdded({ id, imageDTO, pos: { x: bbox.x - layer.x, y: bbox.y - layer.y } }));
    },
  });
};
