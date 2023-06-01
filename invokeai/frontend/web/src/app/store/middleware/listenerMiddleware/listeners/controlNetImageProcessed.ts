import { startAppListening } from '..';
import { imageMetadataReceived, imageUploaded } from 'services/thunks/image';
import { addToast } from 'features/system/store/systemSlice';
import { log } from 'app/logging/useLogger';
import { controlNetImageProcessed } from 'features/controlNet/store/actions';
import { Graph } from 'services/api';
import { sessionCreated } from 'services/thunks/session';
import { sessionReadyToInvoke } from 'features/system/store/actions';
import { appSocketInvocationComplete } from 'services/events/actions';
import { isImageOutput } from 'services/types/guards';
import { controlNetProcessedImageChanged } from 'features/controlNet/store/controlNetSlice';
import { selectImagesById } from 'features/gallery/store/imagesSlice';

const moduleLog = log.child({ namespace: 'controlNet' });

export const addControlNetImageProcessedListener = () => {
  startAppListening({
    actionCreator: controlNetImageProcessed,
    effect: async (action, { dispatch, getState, take }) => {
      const { controlNetId, processorNode } = action.payload;
      const { id } = processorNode;
      const graph: Graph = {
        nodes: { [id]: processorNode },
      };
      const sessionCreatedAction = dispatch(sessionCreated({ graph }));
      const [sessionCreatedFulfilledAction] = await take(
        (action): action is ReturnType<typeof sessionCreated.fulfilled> =>
          sessionCreated.fulfilled.match(action) &&
          action.meta.requestId === sessionCreatedAction.requestId
      );
      const sessionId = sessionCreatedFulfilledAction.payload.id;
      dispatch(sessionReadyToInvoke());
      const [processorAction] = await take(
        (action): action is ReturnType<typeof appSocketInvocationComplete> =>
          appSocketInvocationComplete.match(action) &&
          action.payload.data.graph_execution_state_id === sessionId
      );

      if (isImageOutput(processorAction.payload.data.result)) {
        const { image_name } = processorAction.payload.data.result.image;

        const [imageMetadataReceivedAction] = await take(
          (
            action
          ): action is ReturnType<typeof imageMetadataReceived.fulfilled> =>
            imageMetadataReceived.fulfilled.match(action) &&
            action.payload.image_name === image_name
        );

        const processedControlImage = imageMetadataReceivedAction.payload;
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
