import { log } from 'app/logging/useLogger';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import { controlNetImageChanged } from 'features/controlNet/store/controlNetSlice';
import {
  ASSETS_CATEGORIES,
  INITIAL_IMAGE_LIMIT,
  imagesAddedToBatch,
} from 'features/gallery/store/gallerySlice';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import { initialImageChanged } from 'features/parameters/store/generationSlice';
import { addToast } from 'features/system/store/systemSlice';
import { imageUploaded } from 'services/api/thunks/image';
import { startAppListening } from '..';
import {
  imagesAdapter,
  imagesApi,
} from '../../../../../services/api/endpoints/images';

const moduleLog = log.child({ namespace: 'image' });

export const addImageUploadedFulfilledListener = () => {
  startAppListening({
    actionCreator: imageUploaded.fulfilled,
    effect: (action, { dispatch, getState }) => {
      const image = action.payload;

      moduleLog.debug({ arg: '<Blob>', image }, 'Image uploaded');

      if (action.payload.is_intermediate) {
        // No further actions needed for intermediate images
        return;
      }

      const { boardId } = action.meta.arg;

      // add image to the board
      if (boardId && !['all', 'none', 'batch'].includes(boardId)) {
        dispatch(
          imagesApi.endpoints.addImageToBoard.initiate({
            board_id: boardId,
            imageDTO: image,
          })
        );
        // update board asset list
        const queryArg = {
          categories: ASSETS_CATEGORIES,
          board_id: boardId === 'all' ? undefined : boardId,
          offset: 0,
          limit: INITIAL_IMAGE_LIMIT,
          is_intermediate: false,
        };
        dispatch(
          imagesApi.util.updateQueryData('listImages', queryArg, (draft) => {
            imagesAdapter.addOne(draft, image);
            draft.total = draft.total + 1;
          })
        );
      }

      // update the main list
      const queryArg = {
        categories: ASSETS_CATEGORIES,
      };

      dispatch(
        imagesApi.util.updateQueryData('listImages', queryArg, (draft) => {
          imagesAdapter.addOne(draft, image);
          draft.total = draft.total + 1;
        })
      );

      const { postUploadAction } = action.meta.arg;

      if (postUploadAction?.type === 'TOAST_CANVAS_SAVED_TO_GALLERY') {
        dispatch(
          addToast({ title: 'Canvas Saved to Gallery', status: 'success' })
        );
        return;
      }

      if (postUploadAction?.type === 'TOAST_CANVAS_MERGED') {
        dispatch(addToast({ title: 'Canvas Merged', status: 'success' }));
        return;
      }

      if (postUploadAction?.type === 'SET_CANVAS_INITIAL_IMAGE') {
        dispatch(setInitialCanvasImage(image));
        return;
      }

      if (postUploadAction?.type === 'SET_CONTROLNET_IMAGE') {
        const { controlNetId } = postUploadAction;
        dispatch(
          controlNetImageChanged({
            controlNetId,
            controlImage: image.image_name,
          })
        );
        return;
      }

      if (postUploadAction?.type === 'SET_INITIAL_IMAGE') {
        dispatch(initialImageChanged(image));
        return;
      }

      if (postUploadAction?.type === 'SET_NODES_IMAGE') {
        const { nodeId, fieldName } = postUploadAction;
        dispatch(fieldValueChanged({ nodeId, fieldName, value: image }));
        return;
      }

      if (postUploadAction?.type === 'TOAST_UPLOADED') {
        dispatch(addToast({ title: 'Image Uploaded', status: 'success' }));
        return;
      }

      if (postUploadAction?.type === 'ADD_TO_BATCH') {
        dispatch(imagesAddedToBatch([image.image_name]));
        return;
      }
    },
  });
};

export const addImageUploadedRejectedListener = () => {
  startAppListening({
    actionCreator: imageUploaded.rejected,
    effect: (action, { dispatch }) => {
      const { formData, ...rest } = action.meta.arg;
      const sanitizedData = { arg: { ...rest, formData: { file: '<Blob>' } } };
      moduleLog.error({ data: sanitizedData }, 'Image upload failed');
      dispatch(
        addToast({
          title: 'Image Upload Failed',
          description: action.error.message,
          status: 'error',
        })
      );
    },
  });
};
