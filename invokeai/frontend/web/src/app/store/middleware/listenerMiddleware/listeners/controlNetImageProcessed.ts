import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import { controlNetImageProcessed } from 'features/controlNet/store/actions';
import { controlNetProcessedImageChanged } from 'features/controlNet/store/controlNetSlice';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';
import { imagesApi } from 'services/api/endpoints/images';
import { queueApi } from 'services/api/endpoints/queue';
import { isImageOutput } from 'services/api/guards';
import { Graph, ImageDTO } from 'services/api/types';
import { socketInvocationComplete } from 'services/events/actions';
import { startAppListening } from '..';

export const addControlNetImageProcessedListener = () => {
  startAppListening({
    actionCreator: controlNetImageProcessed,
    effect: async (action, { dispatch, getState, take }) => {
      const log = logger('session');
      const { controlNetId } = action.payload;
      const controlNet = getState().controlNet.controlNets[controlNetId];

      if (!controlNet?.controlImage) {
        log.error('Unable to process ControlNet image');
        return;
      }

      // ControlNet one-off procressing graph is just the processor node, no edges.
      // Also we need to grab the image.
      const graph: Graph = {
        nodes: {
          [controlNet.processorNode.id]: {
            ...controlNet.processorNode,
            is_intermediate: true,
            image: { image_name: controlNet.controlImage },
          },
        },
      };
      try {
        const req = dispatch(
          queueApi.endpoints.enqueueGraph.initiate(
            { graph, prepend: true },
            {
              fixedCacheKey: 'enqueueGraph',
            }
          )
        );
        const enqueueResult = await req.unwrap();
        req.reset();
        dispatch(
          queueApi.endpoints.resumeProcessor.initiate(undefined, {
            fixedCacheKey: 'startQueue',
          })
        );
        log.debug(
          { enqueueResult: parseify(enqueueResult) },
          t('queue.graphQueued')
        );

        const [invocationCompleteAction] = await take(
          (action): action is ReturnType<typeof socketInvocationComplete> =>
            socketInvocationComplete.match(action) &&
            action.payload.data.graph_execution_state_id ===
              enqueueResult.queue_item.session_id
        );

        // We still have to check the output type
        if (isImageOutput(invocationCompleteAction.payload.data.result)) {
          const { image_name } =
            invocationCompleteAction.payload.data.result.image;

          // Wait for the ImageDTO to be received
          const [{ payload }] = await take(
            (action) =>
              imagesApi.endpoints.getImageDTO.matchFulfilled(action) &&
              action.payload.image_name === image_name
          );

          const processedControlImage = payload as ImageDTO;

          log.debug(
            { controlNetId: action.payload, processedControlImage },
            'ControlNet image processed'
          );

          // Update the processed image in the store
          dispatch(
            controlNetProcessedImageChanged({
              controlNetId,
              processedControlImage: processedControlImage.image_name,
            })
          );
        }
      } catch {
        log.error({ graph: parseify(graph) }, t('queue.graphFailedToQueue'));
        dispatch(
          addToast({
            title: t('queue.graphFailedToQueue'),
            status: 'error',
          })
        );
      }
    },
  });
};
