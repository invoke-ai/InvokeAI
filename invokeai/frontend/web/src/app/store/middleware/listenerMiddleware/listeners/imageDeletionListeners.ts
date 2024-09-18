import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import type { AppDispatch, RootState } from 'app/store/store';
import { entityDeleted, referenceImageIPAdapterImageChanged } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import { imageDeletionConfirmed } from 'features/deleteImageModal/store/actions';
import { isModalOpenChanged } from 'features/deleteImageModal/store/slice';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { fieldImageValueChanged } from 'features/nodes/store/nodesSlice';
import { isImageFieldInputInstance } from 'features/nodes/types/field';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { forEach, intersectionBy } from 'lodash-es';
import { imagesApi } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

const log = logger('gallery');

//TODO(psyche): handle image deletion (canvas staging area?)

// Some utils to delete images from different parts of the app
const deleteNodesImages = (state: RootState, dispatch: AppDispatch, imageDTO: ImageDTO) => {
  state.nodes.present.nodes.forEach((node) => {
    if (!isInvocationNode(node)) {
      return;
    }

    forEach(node.data.inputs, (input) => {
      if (isImageFieldInputInstance(input) && input.value?.image_name === imageDTO.image_name) {
        dispatch(
          fieldImageValueChanged({
            nodeId: node.data.id,
            fieldName: input.name,
            value: undefined,
          })
        );
      }
    });
  });
};

const deleteControlLayerImages = (state: RootState, dispatch: AppDispatch, imageDTO: ImageDTO) => {
  selectCanvasSlice(state).controlLayers.entities.forEach(({ id, objects }) => {
    let shouldDelete = false;
    for (const obj of objects) {
      if (obj.type === 'image' && obj.image.image_name === imageDTO.image_name) {
        shouldDelete = true;
        break;
      }
    }
    if (shouldDelete) {
      dispatch(entityDeleted({ entityIdentifier: { id, type: 'control_layer' } }));
    }
  });
};

const deleteReferenceImages = (state: RootState, dispatch: AppDispatch, imageDTO: ImageDTO) => {
  selectCanvasSlice(state).referenceImages.entities.forEach((entity) => {
    if (entity.ipAdapter.image?.image_name === imageDTO.image_name) {
      dispatch(referenceImageIPAdapterImageChanged({ entityIdentifier: getEntityIdentifier(entity), imageDTO: null }));
    }
  });
};

const deleteRasterLayerImages = (state: RootState, dispatch: AppDispatch, imageDTO: ImageDTO) => {
  selectCanvasSlice(state).rasterLayers.entities.forEach(({ id, objects }) => {
    let shouldDelete = false;
    for (const obj of objects) {
      if (obj.type === 'image' && obj.image.image_name === imageDTO.image_name) {
        shouldDelete = true;
        break;
      }
    }
    if (shouldDelete) {
      dispatch(entityDeleted({ entityIdentifier: { id, type: 'raster_layer' } }));
    }
  });
};

export const addImageDeletionListeners = (startAppListening: AppStartListening) => {
  // Handle single image deletion
  startAppListening({
    actionCreator: imageDeletionConfirmed,
    effect: async (action, { dispatch, getState }) => {
      const { imageDTOs, imagesUsage } = action.payload;

      if (imageDTOs.length !== 1 || imagesUsage.length !== 1) {
        // handle multiples in separate listener
        return;
      }

      const imageDTO = imageDTOs[0];
      const imageUsage = imagesUsage[0];

      if (!imageDTO || !imageUsage) {
        // satisfy noUncheckedIndexedAccess
        return;
      }

      try {
        const state = getState();
        await dispatch(imagesApi.endpoints.deleteImage.initiate(imageDTO)).unwrap();

        if (state.gallery.selection.some((i) => i.image_name === imageDTO.image_name)) {
          // The deleted image was a selected image, we need to select the next image
          const newSelection = state.gallery.selection.filter((i) => i.image_name !== imageDTO.image_name);

          if (newSelection.length > 0) {
            return;
          }

          // Get the current list of images and select the same index
          const baseQueryArgs = selectListImagesQueryArgs(state);
          const data = imagesApi.endpoints.listImages.select(baseQueryArgs)(state).data;

          if (data) {
            const deletedImageIndex = data.items.findIndex((i) => i.image_name === imageDTO.image_name);
            const nextImage = data.items[deletedImageIndex + 1] ?? data.items[0] ?? null;
            if (nextImage?.image_name === imageDTO.image_name) {
              // If the next image is the same as the deleted one, it means it was the last image, reset selection
              dispatch(imageSelected(null));
            } else {
              dispatch(imageSelected(nextImage));
            }
          }
        }

        deleteNodesImages(state, dispatch, imageDTO);
        deleteReferenceImages(state, dispatch, imageDTO);
        deleteRasterLayerImages(state, dispatch, imageDTO);
        deleteControlLayerImages(state, dispatch, imageDTO);
      } catch {
        // no-op
      } finally {
        dispatch(isModalOpenChanged(false));
      }
    },
  });

  // Handle multiple image deletion
  startAppListening({
    actionCreator: imageDeletionConfirmed,
    effect: async (action, { dispatch, getState }) => {
      const { imageDTOs, imagesUsage } = action.payload;

      if (imageDTOs.length <= 1 || imagesUsage.length <= 1) {
        // handle singles in separate listener
        return;
      }

      try {
        const state = getState();
        await dispatch(imagesApi.endpoints.deleteImages.initiate({ imageDTOs })).unwrap();

        if (intersectionBy(state.gallery.selection, imageDTOs, 'image_name').length > 0) {
          // Some selected images were deleted, need to select the next image
          const queryArgs = selectListImagesQueryArgs(state);
          const { data } = imagesApi.endpoints.listImages.select(queryArgs)(state);
          if (data) {
            // When we delete multiple images, we clear the selection. Then, the the next time we load images, we will
            // select the first one. This is handled below in the listener for `imagesApi.endpoints.listImages.matchFulfilled`.
            dispatch(imageSelected(null));
          }
        }

        // We need to reset the features where the image is in use - none of these work if their image(s) don't exist

        imageDTOs.forEach((imageDTO) => {
          deleteNodesImages(state, dispatch, imageDTO);
          deleteControlLayerImages(state, dispatch, imageDTO);
          deleteReferenceImages(state, dispatch, imageDTO);
          deleteRasterLayerImages(state, dispatch, imageDTO);
        });
      } catch {
        // no-op
      } finally {
        dispatch(isModalOpenChanged(false));
      }
    },
  });

  // When we list images, if no images is selected, select the first one.
  startAppListening({
    matcher: imagesApi.endpoints.listImages.matchFulfilled,
    effect: (action, { dispatch, getState }) => {
      const selection = getState().gallery.selection;
      if (selection.length === 0) {
        dispatch(imageSelected(action.payload.items[0] ?? null));
      }
    },
  });

  startAppListening({
    matcher: imagesApi.endpoints.deleteImage.matchFulfilled,
    effect: (action) => {
      log.debug({ imageDTO: action.meta.arg.originalArgs }, 'Image deleted');
    },
  });

  startAppListening({
    matcher: imagesApi.endpoints.deleteImage.matchRejected,
    effect: (action) => {
      log.debug({ imageDTO: action.meta.arg.originalArgs }, 'Unable to delete image');
    },
  });
};
