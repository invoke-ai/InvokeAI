import { invocationComplete } from 'services/events/actions';
import { isImageOutput } from 'services/types/guards';
import {
  imageMetadataReceived,
  imageUrlsReceived,
} from 'services/thunks/image';
import { startAppListening } from '..';

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

      const { data, shouldFetchImages } = action.payload;
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

        // const [x] = await take(
        //   (
        //     action
        //   ): action is ReturnType<typeof imageMetadataReceived.fulfilled> =>
        //     imageMetadataReceived.fulfilled.match(action) &&
        //     action.payload.image_name === name
        // );

        // console.log(x);

        // const state = getState();

        // // if we need to refetch, set URLs to placeholder for now
        // const { url, thumbnail } = shouldFetchImages
        //   ? { url: '', thumbnail: '' }
        //   : buildImageUrls(type, name);

        // const timestamp = extractTimestampFromImageName(name);

        // const image: Image = {
        //   name,
        //   type,
        //   url,
        //   thumbnail,
        //   metadata: {
        //     created: timestamp,
        //     width: result.width,
        //     height: result.height,
        //     invokeai: {
        //       session_id: graph_execution_state_id,
        //       ...(node ? { node } : {}),
        //     },
        //   },
        // };

        // dispatch(resultAdded(image));

        // if (state.gallery.shouldAutoSwitchToNewImages) {
        //   dispatch(imageSelected(image));
        // }

        // if (state.config.shouldFetchImages) {
        //   dispatch(imageReceived({ imageName: name, imageType: type }));
        //   dispatch(
        //     thumbnailReceived({
        //       thumbnailName: name,
        //       thumbnailType: type,
        //     })
        //   );
        // }

        // if (
        //   graph_execution_state_id ===
        //   state.canvas.layerState.stagingArea.sessionId
        // ) {
        //   dispatch(addImageToStagingArea(image));
        // }
      }
    },
  });
};
