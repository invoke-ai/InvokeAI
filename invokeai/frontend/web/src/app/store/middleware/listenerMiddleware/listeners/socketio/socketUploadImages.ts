import { logger } from 'app/logging/logger';
import {
  addUploadedImages,
  resetUploadedImages,
} from 'features/gallery/store/gallerySlice';
import { boardsApi } from 'services/api/endpoints/boards';
import { imagesApi } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { getCategories, imagesAdapter } from 'services/api/util';
import { socketUploadImages } from 'services/events/actions';

import { startAppListening } from '../..';

const log = logger('socketio');

export const addSocketUploadImagesEventListener = (): void => {
  startAppListening({
    actionCreator: socketUploadImages,
    effect: (action, { dispatch }) => {
      const { status, images_DTOs, message } = action.payload.data;
      // TODO: update images in gallery based on state
      // TODO: clear state in a good way
      // TODO: handle errors
      // TODO: add dispatching for the toast - add it to the gallery state
      // TODO: add progression for the toast to the gallery state
      // TODO: add errors to the gallery state
      if (status === 'processing' && images_DTOs) {
        // Assuming images_DTOs is an array of ImageDTOs
        images_DTOs.forEach((imageDTO: ImageDTO) => {
          const boardId = imageDTO.board_id ?? 'none';

          // Dispatch to add the image name to the gallery state
          dispatch(addUploadedImages([imageDTO.image_name]));

          // Update `getImageDTO`
          dispatch(
            imagesApi.util.upsertQueryData(
              'getImageDTO',
              imageDTO.image_name,
              imageDTO
            )
          );

          // Update listImages cache if necessary
          const categories = getCategories(imageDTO) as (
            | 'mask'
            | 'general'
            | 'control'
            | 'user'
            | 'other'
          )[]; // Ensure this casting matches your actual categories
          dispatch(
            imagesApi.util.updateQueryData(
              'listImages',
              { board_id: boardId, categories },
              (draft) => {
                imagesAdapter.addOne(draft, imageDTO);
              }
            )
          );

          // Update the board's total image count and invalidate tags
          dispatch(
            boardsApi.util.updateQueryData(
              'getBoardAssetsTotal',
              boardId,
              (draft) => {
                draft.total += 1;
              }
            )
          );

          // Invalidate the tags for updated board totals
          dispatch(
            boardsApi.util.invalidateTags([
              { type: 'BoardAssetsTotal', id: boardId },
              { type: 'BoardImagesTotal', id: boardId },
            ])
          );
        });
      }

      if (status === 'done') {
        // Handle completion logic
        dispatch(resetUploadedImages());
      } else if (status === 'error') {
        log.error(`Upload error: ${message}`);
      }
    },
  });
};
