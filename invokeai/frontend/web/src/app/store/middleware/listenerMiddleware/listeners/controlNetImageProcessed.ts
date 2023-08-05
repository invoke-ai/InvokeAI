import { logger } from 'app/logging/logger';
import { controlNetImageProcessed } from 'features/controlNet/store/actions';
import { controlNetProcessedImageChanged } from 'features/controlNet/store/controlNetSlice';
import { sessionReadyToInvoke } from 'features/system/store/actions';
import { imagesApi } from 'services/api/endpoints/images';
import { isImageOutput } from 'services/api/guards';
import { sessionCreated } from 'services/api/thunks/session';
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

      // Create a session to run the graph & wait til it's ready to invoke
      const sessionCreatedAction = dispatch(sessionCreated({ graph }));
      const [sessionCreatedFulfilledAction] = await take(
        (action): action is ReturnType<typeof sessionCreated.fulfilled> =>
          sessionCreated.fulfilled.match(action) &&
          action.meta.requestId === sessionCreatedAction.requestId
      );

      const sessionId = sessionCreatedFulfilledAction.payload.id;

      // Invoke the session & wait til it's complete
      dispatch(sessionReadyToInvoke());
      const [invocationCompleteAction] = await take(
        (action): action is ReturnType<typeof socketInvocationComplete> =>
          socketInvocationComplete.match(action) &&
          action.payload.data.graph_execution_state_id === sessionId
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
    },
  });
};
