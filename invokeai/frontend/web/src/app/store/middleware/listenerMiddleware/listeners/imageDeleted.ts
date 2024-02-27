import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { resetCanvas } from 'features/canvas/store/canvasSlice';
import {
  controlAdapterImageChanged,
  controlAdapterProcessedImageChanged,
  selectControlAdapterAll,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { isControlNetOrT2IAdapter } from 'features/controlAdapters/store/types';
import { imageDeletionConfirmed } from 'features/deleteImageModal/store/actions';
import { isModalOpenChanged } from 'features/deleteImageModal/store/slice';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { fieldImageValueChanged } from 'features/nodes/store/nodesSlice';
import { isImageFieldInputInstance } from 'features/nodes/types/field';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { clearInitialImage } from 'features/parameters/store/generationSlice';
import { clamp, forEach } from 'lodash-es';
import { api } from 'services/api';
import { imagesApi } from 'services/api/endpoints/images';
import { imagesSelectors } from 'services/api/util';

export const addRequestedSingleImageDeletionListener = (startAppListening: AppStartListening) => {
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
      const lastSelectedImage = state.gallery.selection[state.gallery.selection.length - 1]?.image_name;

      if (imageDTO && imageDTO?.image_name === lastSelectedImage) {
        const { image_name } = imageDTO;

        const baseQueryArgs = selectListImagesQueryArgs(state);
        const { data } = imagesApi.endpoints.listImages.select(baseQueryArgs)(state);

        const cachedImageDTOs = data ? imagesSelectors.selectAll(data) : [];

        const deletedImageIndex = cachedImageDTOs.findIndex((i) => i.image_name === image_name);

        const filteredImageDTOs = cachedImageDTOs.filter((i) => i.image_name !== image_name);

        const newSelectedImageIndex = clamp(deletedImageIndex, 0, filteredImageDTOs.length - 1);

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
        if (getState().generation.initialImage?.imageName === imageDTO.image_name) {
          dispatch(clearInitialImage());
        }

        // reset control adapters that use the deleted images
        forEach(selectControlAdapterAll(getState().controlAdapters), (ca) => {
          if (
            ca.controlImage === imageDTO.image_name ||
            (isControlNetOrT2IAdapter(ca) && ca.processedControlImage === imageDTO.image_name)
          ) {
            dispatch(
              controlAdapterImageChanged({
                id: ca.id,
                controlImage: null,
              })
            );
            dispatch(
              controlAdapterProcessedImageChanged({
                id: ca.id,
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
      });

      // Delete from server
      const { requestId } = dispatch(imagesApi.endpoints.deleteImage.initiate(imageDTO));

      // Wait for successful deletion, then trigger boards to re-fetch
      const wasImageDeleted = await condition(
        (action) => imagesApi.endpoints.deleteImage.matchFulfilled(action) && action.meta.requestId === requestId,
        30000
      );

      if (wasImageDeleted) {
        dispatch(api.util.invalidateTags([{ type: 'Board', id: imageDTO.board_id ?? 'none' }]));
      }
    },
  });

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
        await dispatch(imagesApi.endpoints.deleteImages.initiate({ imageDTOs })).unwrap();
        const state = getState();
        const queryArgs = selectListImagesQueryArgs(state);
        const { data } = imagesApi.endpoints.listImages.select(queryArgs)(state);

        const newSelectedImageDTO = data ? imagesSelectors.selectAll(data)[0] : undefined;

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
          if (getState().generation.initialImage?.imageName === imageDTO.image_name) {
            dispatch(clearInitialImage());
          }

          // reset control adapters that use the deleted images
          forEach(selectControlAdapterAll(getState().controlAdapters), (ca) => {
            if (
              ca.controlImage === imageDTO.image_name ||
              (isControlNetOrT2IAdapter(ca) && ca.processedControlImage === imageDTO.image_name)
            ) {
              dispatch(
                controlAdapterImageChanged({
                  id: ca.id,
                  controlImage: null,
                })
              );
              dispatch(
                controlAdapterProcessedImageChanged({
                  id: ca.id,
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
        });
      } catch {
        // no-op
      }
    },
  });

  startAppListening({
    matcher: imagesApi.endpoints.deleteImage.matchPending,
    effect: () => {
      //
    },
  });

  startAppListening({
    matcher: imagesApi.endpoints.deleteImage.matchFulfilled,
    effect: (action) => {
      const log = logger('images');
      log.debug({ imageDTO: action.meta.arg.originalArgs }, 'Image deleted');
    },
  });

  startAppListening({
    matcher: imagesApi.endpoints.deleteImage.matchRejected,
    effect: (action) => {
      const log = logger('images');
      log.debug({ imageDTO: action.meta.arg.originalArgs }, 'Unable to delete image');
    },
  });
};
