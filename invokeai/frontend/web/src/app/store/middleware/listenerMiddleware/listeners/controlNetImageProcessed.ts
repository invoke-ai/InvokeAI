import { startAppListening } from '..';
import { imageMetadataReceived } from 'services/thunks/image';
import { log } from 'app/logging/useLogger';
import { controlNetImageProcessed } from 'features/controlNet/store/actions';
import { Graph } from 'services/api';
import { sessionCreated } from 'services/thunks/session';
import { sessionReadyToInvoke } from 'features/system/store/actions';
import { socketInvocationComplete } from 'services/events/actions';
import { isImageOutput } from 'services/types/guards';
import { controlNetProcessedImageChanged } from 'features/controlNet/store/controlNetSlice';

const moduleLog = log.child({ namespace: 'controlNet' });

export const addControlNetImageProcessedListener = () => {
  startAppListening({
    actionCreator: controlNetImageProcessed,
    effect: async (action, { dispatch, getState, take }) => {
      const { controlNetId, processorNode } = action.payload;

      // ControlNet one-off procressing graph is just he processor node, no edges
      const graph: Graph = {
        nodes: { [processorNode.id]: processorNode },
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
        const [imageMetadataReceivedAction] = await take(
          (
            action
          ): action is ReturnType<typeof imageMetadataReceived.fulfilled> =>
            imageMetadataReceived.fulfilled.match(action) &&
            action.payload.image_name === image_name
        );
        const processedControlImage = imageMetadataReceivedAction.payload;

        // Update the processed image in the store
        dispatch(
          controlNetProcessedImageChanged({
            controlNetId,
            processedControlImage,
          })
        );
      }
    },
  });
};
