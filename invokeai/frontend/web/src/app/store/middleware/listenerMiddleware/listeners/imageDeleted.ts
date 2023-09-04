import { logger } from 'app/logging/logger';
import { resetCanvas } from 'features/canvas/store/canvasSlice';
import {
  controlNetImageChanged,
  controlNetProcessedImageChanged,
} from 'features/controlNet/store/controlNetSlice';
import { imageDeletionConfirmed } from 'features/deleteImageModal/store/actions';
import { isModalOpenChanged } from 'features/deleteImageModal/store/slice';
import { selectListImagesBaseQueryArgs } from 'features/gallery/store/gallerySelectors';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { fieldImageValueChanged } from 'features/nodes/store/nodesSlice';
import { isInvocationNode } from 'features/nodes/types/types';
import { clearInitialImage } from 'features/parameters/store/generationSlice';
import { clamp, forEach } from 'lodash-es';
import { api } from 'services/api';
import { imagesApi } from 'services/api/endpoints/images';
import { imagesAdapter } from 'services/api/util';
import { startAppListening } from '..';

export const addRequestedSingleImageDeletionListener = () => {
  startAppListening({
    actionCreator: imageDeletionConfirmed,
    effect: async (action, { dispatch, getState, condition }) => {
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

      dispatch(isModalOpenChanged(false));

      const state = getState();
      const lastSelectedImage =
        state.gallery.selection[state.gallery.selection.length - 1]?.image_name;

      if (imageDTO && imageDTO?.image_name === lastSelectedImage) {
        const { image_name } = imageDTO;

        const baseQueryArgs = selectListImagesBaseQueryArgs(state);
        const { data } =
          imagesApi.endpoints.listImages.select(baseQueryArgs)(state);

        const cachedImageDTOs = data
          ? imagesAdapter.getSelectors().selectAll(data)
          : [];

        const deletedImageIndex = cachedImageDTOs.findIndex(
          (i) => i.image_name === image_name
        );

        const filteredImageDTOs = cachedImageDTOs.filter(
          (i) => i.image_name !== image_name
        );

        const newSelectedImageIndex = clamp(
          deletedImageIndex,
          0,
          filteredImageDTOs.length - 1
        );

        const newSelectedImageDTO = filteredImageDTOs[newSelectedImageIndex];

        if (newSelectedImageDTO) {
          dispatch(imageSelected(newSelectedImageDTO));
        } else {
          dispatch(imageSelected(null));
        }
      }

      // We need to reset the features where the image is in use - none of these work if their image(s) don't exist
      if (imageUsage.isCanvasImage) {
        dispatch(resetCanvas());
      }

      imageDTOs.forEach((imageDTO) => {
        // reset init image if we deleted it
        if (
          getState().generation.initialImage?.imageName === imageDTO.image_name
        ) {
          dispatch(clearInitialImage());
        }

        // reset controlNets that use the deleted images
        forEach(getState().controlNet.controlNets, (controlNet) => {
          if (
            controlNet.controlImage === imageDTO.image_name ||
            controlNet.processedControlImage === imageDTO.image_name
          ) {
            dispatch(
              controlNetImageChanged({
                controlNetId: controlNet.controlNetId,
                controlImage: null,
              })
            );
            dispatch(
              controlNetProcessedImageChanged({
                controlNetId: controlNet.controlNetId,
                processedControlImage: null,
              })
            );
          }
        });

        // reset nodes that use the deleted images
        getState().nodes.nodes.forEach((node) => {
          if (!isInvocationNode(node)) {
            return;
          }

          forEach(node.data.inputs, (input) => {
            if (
              input.type === 'ImageField' &&
              input.value?.image_name === imageDTO.image_name
            ) {
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
      });

      // Delete from server
      const { requestId } = dispatch(
        imagesApi.endpoints.deleteImage.initiate(imageDTO)
      );

      // Wait for successful deletion, then trigger boards to re-fetch
      const wasImageDeleted = await condition(
        (action) =>
          imagesApi.endpoints.deleteImage.matchFulfilled(action) &&
          action.meta.requestId === requestId,
        30000
      );

      if (wasImageDeleted) {
        dispatch(
          api.util.invalidateTags([{ type: 'Board', id: imageDTO.board_id }])
        );
      }
    },
  });
};

/**
 * Called when the user requests an image deletion
 */
export const addRequestedMultipleImageDeletionListener = () => {
  startAppListening({
    actionCreator: imageDeletionConfirmed,
    effect: async (action, { dispatch, getState }) => {
      const { imageDTOs, imagesUsage } = action.payload;

      if (imageDTOs.length <= 1 || imagesUsage.length <= 1) {
        // handle singles in separate listener
        return;
      }

      try {
        // Delete from server
        await dispatch(
          imagesApi.endpoints.deleteImages.initiate({ imageDTOs })
        ).unwrap();
        const state = getState();
        const baseQueryArgs = selectListImagesBaseQueryArgs(state);
        const { data } =
          imagesApi.endpoints.listImages.select(baseQueryArgs)(state);

        const newSelectedImageDTO = data
          ? imagesAdapter.getSelectors().selectAll(data)[0]
          : undefined;

        if (newSelectedImageDTO) {
          dispatch(imageSelected(newSelectedImageDTO));
        } else {
          dispatch(imageSelected(null));
        }

        dispatch(isModalOpenChanged(false));

        // We need to reset the features where the image is in use - none of these work if their image(s) don't exist

        if (imagesUsage.some((i) => i.isCanvasImage)) {
          dispatch(resetCanvas());
        }

        imageDTOs.forEach((imageDTO) => {
          // reset init image if we deleted it
          if (
            getState().generation.initialImage?.imageName ===
            imageDTO.image_name
          ) {
            dispatch(clearInitialImage());
          }

          // reset controlNets that use the deleted images
          forEach(getState().controlNet.controlNets, (controlNet) => {
            if (
              controlNet.controlImage === imageDTO.image_name ||
              controlNet.processedControlImage === imageDTO.image_name
            ) {
              dispatch(
                controlNetImageChanged({
                  controlNetId: controlNet.controlNetId,
                  controlImage: null,
                })
              );
              dispatch(
                controlNetProcessedImageChanged({
                  controlNetId: controlNet.controlNetId,
                  processedControlImage: null,
                })
              );
            }
          });

          // reset nodes that use the deleted images
          getState().nodes.nodes.forEach((node) => {
            if (!isInvocationNode(node)) {
              return;
            }

            forEach(node.data.inputs, (input) => {
              if (
                input.type === 'ImageField' &&
                input.value?.image_name === imageDTO.image_name
              ) {
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
        });
      } catch {
        // no-op
      }
    },
  });
};

/**
 * Called when the actual delete request is sent to the server
 */
export const addImageDeletedPendingListener = () => {
  startAppListening({
    matcher: imagesApi.endpoints.deleteImage.matchPending,
    effect: () => {
      //
    },
  });
};

/**
 * Called on successful delete
 */
export const addImageDeletedFulfilledListener = () => {
  startAppListening({
    matcher: imagesApi.endpoints.deleteImage.matchFulfilled,
    effect: (action) => {
      const log = logger('images');
      log.debug({ imageDTO: action.meta.arg.originalArgs }, 'Image deleted');
    },
  });
};

/**
 * Called on failed delete
 */
export const addImageDeletedRejectedListener = () => {
  startAppListening({
    matcher: imagesApi.endpoints.deleteImage.matchRejected,
    effect: (action) => {
      const log = logger('images');
      log.debug(
        { imageDTO: action.meta.arg.originalArgs },
        'Unable to delete image'
      );
    },
  });
};
