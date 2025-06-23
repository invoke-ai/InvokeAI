import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { imagesApi } from 'services/api/endpoints/images';

export const addImagesStarredListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: imagesApi.endpoints.starImages.matchFulfilled,
    effect: (action, { dispatch, getState }) => {
      // const { updated_image_names: starredImages } = action.payload;
      // const state = getState();
      // const { selection } = state.gallery;
      // const updatedSelection: ImageDTO[] = [];
      // selection.forEach((selectedImageDTO) => {
      //   if (starredImages.includes(selectedImageDTO.image_name)) {
      //     updatedSelection.push({
      //       ...selectedImageDTO,
      //       starred: true,
      //     });
      //   } else {
      //     updatedSelection.push(selectedImageDTO);
      //   }
      // });
      // dispatch(selectionChanged(updatedSelection));
    },
  });
};
