import { isAnyOf } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import {
  sessionStagingAreaImageAccepted,
  sessionStagingAreaReset,
} from 'features/controlLayers/store/canvasSessionSlice';
import { canvasReset, rasterLayerAdded } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/types';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { queueApi } from 'services/api/endpoints/queue';
import { $lastCanvasProgressEvent } from 'services/events/setEventListeners';
import { assert } from 'tsafe';

const log = logger('canvas');

const matchCanvasOrStagingAreaRest = isAnyOf(sessionStagingAreaReset, canvasReset);

export const addStagingListeners = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: matchCanvasOrStagingAreaRest,
    effect: async (_, { dispatch }) => {
      try {
        const req = dispatch(
          queueApi.endpoints.cancelByBatchDestination.initiate(
            { destination: 'canvas' },
            { fixedCacheKey: 'cancelByBatchOrigin' }
          )
        );
        const { canceled } = await req.unwrap();
        req.reset();

        $lastCanvasProgressEvent.set(null);

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
      const stagingAreaImage = state.canvasSession.stagedImages[index];

      assert(stagingAreaImage, 'No staged image found to accept');
      const { x, y } = selectCanvasSlice(state).bbox.rect;

      const { imageDTO, offsetX, offsetY } = stagingAreaImage;
      const imageObject = imageDTOToImageObject(imageDTO);
      const overrides: Partial<CanvasRasterLayerState> = {
        position: { x: x + offsetX, y: y + offsetY },
        objects: [imageObject],
      };

      api.dispatch(rasterLayerAdded({ overrides, isSelected: false }));
      api.dispatch(sessionStagingAreaReset());
    },
  });
};
