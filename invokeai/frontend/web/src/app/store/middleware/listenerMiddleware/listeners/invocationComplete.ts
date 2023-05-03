import { invocationComplete } from 'services/events/actions';
import { isImageOutput } from 'services/types/guards';
import {
  buildImageUrls,
  extractTimestampFromImageName,
} from 'services/util/deserializeImageField';
import { Image } from 'app/types/invokeai';
import { resultAdded } from 'features/gallery/store/resultsSlice';
import { imageReceived, thumbnailReceived } from 'services/thunks/image';
import { startAppListening } from '..';
import { imageSelected } from 'features/gallery/store/gallerySlice';
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
    effect: (action, { getState, dispatch }) => {
      if (!invocationComplete.match(action)) {
        return;
      }

      const { data, shouldFetchImages } = action.payload;
      const { result, node, graph_execution_state_id } = data;

      if (isImageOutput(result) && !nodeDenylist.includes(node.type)) {
        const name = result.image.image_name;
        const type = result.image.image_type;
        const state = getState();

        // if we need to refetch, set URLs to placeholder for now
        const { url, thumbnail } = shouldFetchImages
          ? { url: '', thumbnail: '' }
          : buildImageUrls(type, name);

        const timestamp = extractTimestampFromImageName(name);

        const image: Image = {
          name,
          type,
          url,
          thumbnail,
          metadata: {
            created: timestamp,
            width: result.width,
            height: result.height,
            invokeai: {
              session_id: graph_execution_state_id,
              ...(node ? { node } : {}),
            },
          },
        };

        dispatch(resultAdded(image));

        if (state.gallery.shouldAutoSwitchToNewImages) {
          dispatch(imageSelected(image));
        }

        if (state.config.shouldFetchImages) {
          dispatch(imageReceived({ imageName: name, imageType: type }));
          dispatch(
            thumbnailReceived({
              thumbnailName: name,
              thumbnailType: type,
            })
          );
        }

        if (
          graph_execution_state_id ===
          state.canvas.layerState.stagingArea.sessionId
        ) {
          dispatch(addImageToStagingArea(image));
        }
      }
    },
  });
};
