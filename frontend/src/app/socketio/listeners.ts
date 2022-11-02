import { AnyAction, MiddlewareAPI, Dispatch } from '@reduxjs/toolkit';
import { v4 as uuidv4 } from 'uuid';
import dateFormat from 'dateformat';

import * as InvokeAI from '../invokeai';

import {
  addLogEntry,
  setIsConnected,
  setIsProcessing,
  setSystemStatus,
  setCurrentStatus,
  setSystemConfig,
  processingCanceled,
  errorOccurred,
  setModelList,
  setIsCancelable,
} from '../../features/system/systemSlice';

import {
  addGalleryImages,
  addImage,
  clearIntermediateImage,
  GalleryState,
  removeImage,
  setCurrentImage,
  setIntermediateImage,
} from '../../features/gallery/gallerySlice';

import {
  clearInitialImage,
  setInitialImage,
  setMaskPath,
} from '../../features/options/optionsSlice';
import { requestImages, requestNewImages } from './actions';
import {
  clearImageToInpaint,
  setImageToInpaint,
} from '../../features/tabs/Inpainting/inpaintingSlice';
import { tabMap } from '../../features/tabs/InvokeTabs';

/**
 * Returns an object containing listener callbacks for socketio events.
 * TODO: This file is large, but simple. Should it be split up further?
 */
const makeSocketIOListeners = (
  store: MiddlewareAPI<Dispatch<AnyAction>, any>
) => {
  const { dispatch, getState } = store;

  return {
    /**
     * Callback to run when we receive a 'connect' event.
     */
    onConnect: () => {
      try {
        dispatch(setIsConnected(true));
        dispatch(setCurrentStatus('Connected'));
        const gallery: GalleryState = getState().gallery;

        if (gallery.categories.user.latest_mtime) {
          dispatch(requestNewImages('user'));
        } else {
          dispatch(requestImages('user'));
        }

        if (gallery.categories.result.latest_mtime) {
          dispatch(requestNewImages('result'));
        } else {
          dispatch(requestImages('result'));
        }
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'disconnect' event.
     */
    onDisconnect: () => {
      try {
        dispatch(setIsConnected(false));
        dispatch(setCurrentStatus('Disconnected'));

        dispatch(
          addLogEntry({
            timestamp: dateFormat(new Date(), 'isoDateTime'),
            message: `Disconnected from server`,
            level: 'warning',
          })
        );
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'generationResult' event.
     */
    onGenerationResult: (data: InvokeAI.ImageResultResponse) => {
      try {
        const { shouldLoopback, activeTab } = getState().options;
        const newImage = {
          uuid: uuidv4(),
          ...data,
          category: 'result',
        };

        dispatch(
          addImage({
            category: 'result',
            image: newImage,
          })
        );

        if (shouldLoopback) {
          const activeTabName = tabMap[activeTab];
          switch (activeTabName) {
            case 'img2img': {
              dispatch(setInitialImage(newImage));
              break;
            }
            case 'inpainting': {
              dispatch(setImageToInpaint(newImage));
              break;
            }
          }
        }

        dispatch(
          addLogEntry({
            timestamp: dateFormat(new Date(), 'isoDateTime'),
            message: `Image generated: ${data.url}`,
          })
        );
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'intermediateResult' event.
     */
    onIntermediateResult: (data: InvokeAI.ImageResultResponse) => {
      try {
        dispatch(
          setIntermediateImage({
            uuid: uuidv4(),
            ...data,
          })
        );
        if (!data.isBase64) {
          dispatch(
            addLogEntry({
              timestamp: dateFormat(new Date(), 'isoDateTime'),
              message: `Intermediate image generated: ${data.url}`,
            })
          );
        }
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive an 'esrganResult' event.
     */
    onPostprocessingResult: (data: InvokeAI.ImageResultResponse) => {
      try {
        dispatch(
          addImage({
            category: 'result',
            image: {
              uuid: uuidv4(),
              ...data,
              category: 'result',
            },
          })
        );

        dispatch(
          addLogEntry({
            timestamp: dateFormat(new Date(), 'isoDateTime'),
            message: `Postprocessed: ${data.url}`,
          })
        );
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'progressUpdate' event.
     * TODO: Add additional progress phases
     */
    onProgressUpdate: (data: InvokeAI.SystemStatus) => {
      try {
        dispatch(setIsProcessing(true));
        dispatch(setSystemStatus(data));
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'progressUpdate' event.
     */
    onError: (data: InvokeAI.ErrorResponse) => {
      const { message, additionalData } = data;

      if (additionalData) {
        // TODO: handle more data than short message
      }

      try {
        dispatch(
          addLogEntry({
            timestamp: dateFormat(new Date(), 'isoDateTime'),
            message: `Server error: ${message}`,
            level: 'error',
          })
        );
        dispatch(errorOccurred());
        dispatch(clearIntermediateImage());
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'galleryImages' event.
     */
    onGalleryImages: (data: InvokeAI.GalleryImagesResponse) => {
      const { images, areMoreImagesAvailable, category } = data;

      /**
       * the logic here ideally would be in the reducer but we have a side effect:
       * generating a uuid. so the logic needs to be here, outside redux.
       */

      // Generate a UUID for each image
      const preparedImages = images.map((image): InvokeAI.Image => {
        return {
          uuid: uuidv4(),
          ...image,
        };
      });

      dispatch(
        addGalleryImages({
          images: preparedImages,
          areMoreImagesAvailable,
          category,
        })
      );

      dispatch(
        addLogEntry({
          timestamp: dateFormat(new Date(), 'isoDateTime'),
          message: `Loaded ${images.length} images`,
        })
      );
    },
    /**
     * Callback to run when we receive a 'processingCanceled' event.
     */
    onProcessingCanceled: () => {
      dispatch(processingCanceled());

      const { intermediateImage } = getState().gallery;

      if (intermediateImage) {
        if (!intermediateImage.isBase64) {
          dispatch(
            addImage({
              category: 'result',
              image: intermediateImage,
            })
          );
          dispatch(
            addLogEntry({
              timestamp: dateFormat(new Date(), 'isoDateTime'),
              message: `Intermediate image saved: ${intermediateImage.url}`,
            })
          );
        }
        dispatch(clearIntermediateImage());
      }

      dispatch(
        addLogEntry({
          timestamp: dateFormat(new Date(), 'isoDateTime'),
          message: `Processing canceled`,
          level: 'warning',
        })
      );
    },
    /**
     * Callback to run when we receive a 'imageDeleted' event.
     */
    onImageDeleted: (data: InvokeAI.ImageDeletedResponse) => {
      const { url } = data;

      // remove image from gallery
      dispatch(removeImage(data));

      // remove references to image in options
      const { initialImage, maskPath } = getState().options;
      const { imageToInpaint } = getState().inpainting;

      if (initialImage?.url === url || initialImage === url) {
        dispatch(clearInitialImage());
      }

      if (imageToInpaint?.url === url) {
        dispatch(clearImageToInpaint());
      }

      if (maskPath === url) {
        dispatch(setMaskPath(''));
      }

      dispatch(
        addLogEntry({
          timestamp: dateFormat(new Date(), 'isoDateTime'),
          message: `Image deleted: ${url}`,
        })
      );
    },
    onImageUploaded: (data: InvokeAI.ImageUploadResponse) => {
      const { destination, ...rest } = data;
      const image = {
        uuid: uuidv4(),
        ...rest,
      };

      try {
        dispatch(addImage({ image, category: 'user' }));

        switch (destination) {
          case 'img2img': {
            dispatch(setInitialImage(image));
            break;
          }
          case 'inpainting': {
            dispatch(setImageToInpaint(image));
            break;
          }
          default: {
            dispatch(setCurrentImage(image));
            break;
          }
        }

        dispatch(
          addLogEntry({
            timestamp: dateFormat(new Date(), 'isoDateTime'),
            message: `Image uploaded: ${data.url}`,
          })
        );
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'maskImageUploaded' event.
     */
    onMaskImageUploaded: (data: InvokeAI.ImageUrlResponse) => {
      const { url } = data;
      dispatch(setMaskPath(url));
      dispatch(
        addLogEntry({
          timestamp: dateFormat(new Date(), 'isoDateTime'),
          message: `Mask image uploaded: ${url}`,
        })
      );
    },
    onSystemConfig: (data: InvokeAI.SystemConfig) => {
      dispatch(setSystemConfig(data));
    },
    onModelChanged: (data: InvokeAI.ModelChangeResponse) => {
      const { model_name, model_list } = data;
      dispatch(setModelList(model_list));
      dispatch(setCurrentStatus('Model Changed'));
      dispatch(setIsProcessing(false));
      dispatch(setIsCancelable(true));
      dispatch(
        addLogEntry({
          timestamp: dateFormat(new Date(), 'isoDateTime'),
          message: `Model changed: ${model_name}`,
          level: 'info',
        })
      );
    },
    onModelChangeFailed: (data: InvokeAI.ModelChangeResponse) => {
      const { model_name, model_list } = data;
      dispatch(setModelList(model_list));
      dispatch(setIsProcessing(false));
      dispatch(setIsCancelable(true));
      dispatch(errorOccurred());
      dispatch(
        addLogEntry({
          timestamp: dateFormat(new Date(), 'isoDateTime'),
          message: `Model change failed: ${model_name}`,
          level: 'error',
        })
      );
    },
  };
};

export default makeSocketIOListeners;
