import { AnyAction, Dispatch, MiddlewareAPI } from '@reduxjs/toolkit';
import dateFormat from 'dateformat';
import { Socket } from 'socket.io-client';
import { frontendToBackendParameters } from '../../common/util/parameterTranslation';
import {
  addLogEntry,
  setIsProcessing,
} from '../../features/system/systemSlice';
import { tabMap, tab_dict } from '../../features/tabs/InvokeTabs';
import * as InvokeAI from '../invokeai';
import { GenerateImageProps } from './actions';

/**
 * Returns an object containing all functions which use `socketio.emit()`.
 * i.e. those which make server requests.
 */
const makeSocketIOEmitters = (
  store: MiddlewareAPI<Dispatch<AnyAction>, any>,
  socketio: Socket
) => {
  // We need to dispatch actions to redux and get pieces of state from the store.
  const { dispatch, getState } = store;

  return {
    emitGenerateImage: (args: GenerateImageProps | undefined) => {
      const inpaintingMask = args?.inpaintingMask;

      dispatch(setIsProcessing(true));

      const options = getState().options;

      if (tabMap[options.activeTab] !== 'img2img') {
        options.shouldUseInitImage = false;
      }

      const {
        currentImage: { url: currentImageUrl },
      } = getState().gallery;

      const { generationParameters, esrganParameters, facetoolParameters } =
        frontendToBackendParameters(
          options,
          getState().system,
          tabMap[options.activeTab] as keyof typeof tab_dict,
          inpaintingMask,
          currentImageUrl
        );

      socketio.emit(
        'generateImage',
        generationParameters,
        esrganParameters,
        facetoolParameters
      );

      // we need to truncate the init_mask base64 else it takes up the whole log
      // TODO: handle maintaining masks for reproducibility in future
      if (generationParameters.init_mask) {
        generationParameters.init_mask = generationParameters.init_mask
          .substr(0, 20)
          .concat('...');
      }

      dispatch(
        addLogEntry({
          timestamp: dateFormat(new Date(), 'isoDateTime'),
          message: `Image generation requested: ${JSON.stringify({
            ...generationParameters,
            ...esrganParameters,
            ...facetoolParameters,
          })}`,
        })
      );
    },
    emitOutpaintImage: (args: GenerateImageProps) =>  {
      dispatch(setIsProcessing(true));
      const options = { ...getState().options };
      const imageToProcess = args.inpaintingMask;
      const { generationParameters, esrganParameters, gfpganParameters } =
        frontendToBackendParameters(options, getState().system, 'outpainting');

      socketio.emit('outpaintImage', imageToProcess, {
        type: 'outpaint',
        ...generationParameters,
      });
      dispatch(
        addLogEntry({
          timestamp: dateFormat(new Date(), 'isoDateTime'),
          message: `Outpaint requested: ${JSON.stringify({
            ...generationParameters,
          })}`,
        })
      );
    },
    emitRunESRGAN: (imageToProcess: InvokeAI.Image) => {
      dispatch(setIsProcessing(true));
      const { upscalingLevel, upscalingStrength } = getState().options;
      const esrganParameters = {
        upscale: [upscalingLevel, upscalingStrength],
      };
      socketio.emit('runPostprocessing', imageToProcess, {
        type: 'esrgan',
        ...esrganParameters,
      });
      dispatch(
        addLogEntry({
          timestamp: dateFormat(new Date(), 'isoDateTime'),
          message: `ESRGAN upscale requested: ${JSON.stringify({
            file: imageToProcess.url,
            ...esrganParameters,
          })}`,
        })
      );
    },
    emitRunFacetool: (imageToProcess: InvokeAI.Image) => {
      dispatch(setIsProcessing(true));
      const { facetoolType, facetoolStrength, codeformerFidelity } =
        getState().options;

      const facetoolParameters: Record<string, any> = {
        facetool_strength: facetoolStrength,
      };

      if (facetoolType === 'codeformer') {
        facetoolParameters.codeformer_fidelity = codeformerFidelity;
      }

      socketio.emit('runPostprocessing', imageToProcess, {
        type: facetoolType,
        ...facetoolParameters,
      });
      dispatch(
        addLogEntry({
          timestamp: dateFormat(new Date(), 'isoDateTime'),
          message: `Face restoration (${facetoolType}) requested: ${JSON.stringify(
            {
              file: imageToProcess.url,
              ...facetoolParameters,
            }
          )}`,
        })
      );
    },
    emitDeleteImage: (imageToDelete: InvokeAI.Image) => {
      const { url, uuid } = imageToDelete;
      socketio.emit('deleteImage', url, uuid);
    },
    emitRequestImages: () => {
      const { earliest_mtime } = getState().gallery;
      socketio.emit('requestImages', earliest_mtime);
    },
    emitRequestNewImages: () => {
      const { latest_mtime } = getState().gallery;
      socketio.emit('requestLatestImages', latest_mtime);
    },
    emitCancelProcessing: () => {
      socketio.emit('cancel');
    },
    emitUploadInitialImage: (file: File) => {
      socketio.emit('uploadInitialImage', file, file.name);
    },
    emitUploadMaskImage: (file: File) => {
      socketio.emit('uploadMaskImage', file, file.name);
    },
    emitRequestSystemConfig: () => {
      socketio.emit('requestSystemConfig');
    },
  };
};

export default makeSocketIOEmitters;
