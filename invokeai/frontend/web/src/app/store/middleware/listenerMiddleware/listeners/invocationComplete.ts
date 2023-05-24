import { invocationComplete } from 'services/events/actions';
import { isImageOutput } from 'services/types/guards';
import {
  imageMetadataReceived,
  imageUrlsReceived,
} from 'services/thunks/image';
import { startAppListening } from '..';
import { addImageToStagingArea } from 'features/canvas/store/canvasSlice';

const nodeDenylist = ['dataURL_image'];

export const addImageResultReceivedListener = () => {
  startAppListening({
    predicate: (action) => {
      if (
        invocationComplete.match(action) &&
        isImageOutput(action.payload.data.result)
      ) {
        return true;
      }
      return false;
    },
    effect: async (action, { getState, dispatch, take }) => {
      if (!invocationComplete.match(action)) {
        return;
      }

      const { data } = action.payload;
      const { result, node, graph_execution_state_id } = data;

      if (isImageOutput(result) && !nodeDenylist.includes(node.type)) {
        const { image_name, image_type } = result.image;

        dispatch(
          imageUrlsReceived({ imageName: image_name, imageType: image_type })
        );

        dispatch(
          imageMetadataReceived({
            imageName: image_name,
            imageType: image_type,
          })
        );

        // Handle canvas image
        if (
          graph_execution_state_id ===
          getState().canvas.layerState.stagingArea.sessionId
        ) {
          const [{ payload: image }] = await take(
            (
              action
            ): action is ReturnType<typeof imageMetadataReceived.fulfilled> =>
              imageMetadataReceived.fulfilled.match(action) &&
              action.payload.image_name === image_name
          );
          dispatch(addImageToStagingArea(image));
        }
      }
    },
  });
};
