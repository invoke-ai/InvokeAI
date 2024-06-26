import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import {
  layerAdded,
  layerImageAdded,
  stagingAreaImageAccepted,
  stagingAreaReset,
} from 'features/controlLayers/store/canvasV2Slice';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { queueApi } from 'services/api/endpoints/queue';
import { assert } from 'tsafe';

export const addStagingListeners = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: stagingAreaReset,
    effect: async (_, { dispatch, getState }) => {
      const log = logger('canvas');
      const stagingArea = getState().canvasV2.stagingArea;

      if (!stagingArea) {
        // Should not happen
        return;
      }

      if (stagingArea.batchIds.length === 0) {
        return;
      }

      try {
        const req = dispatch(
          queueApi.endpoints.cancelByBatchIds.initiate(
            { batch_ids: stagingArea.batchIds },
            { fixedCacheKey: 'cancelByBatchIds' }
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
      const { layers, stagingArea, selectedEntityIdentifier } = api.getState().canvasV2;
      let layer = layers.entities.find((layer) => layer.id === selectedEntityIdentifier?.id);

      if (!layer) {
        layer = layers.entities[0];
      }

      if (!layer) {
        // We need to create a new layer to add the accepted image
        api.dispatch(layerAdded());
        layer = layers.entities[0];
      }

      assert(layer, 'No layer found to stage image');
      assert(stagingArea, 'Staging should be defined');

      const { x, y } = stagingArea.bbox;
      const { id } = layer;

      api.dispatch(layerImageAdded({ id, imageDTO, pos: { x, y } }));
      api.dispatch(stagingAreaReset());
    },
  });
};
