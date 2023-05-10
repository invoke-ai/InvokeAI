// import { AnyAction, Dispatch, MiddlewareAPI } from '@reduxjs/toolkit';
// import dateFormat from 'dateformat';
// import i18n from 'i18n';
// import { v4 as uuidv4 } from 'uuid';

// import * as InvokeAI from 'app/types/invokeai';

// import {
//   addToast,
//   errorOccurred,
//   processingCanceled,
//   setCurrentStatus,
//   setFoundModels,
//   setIsCancelable,
//   setIsConnected,
//   setIsProcessing,
//   setModelList,
//   setSearchFolder,
//   setSystemConfig,
//   setSystemStatus,
// } from 'features/system/store/systemSlice';

// import {
//   addGalleryImages,
//   addImage,
//   clearIntermediateImage,
//   GalleryState,
//   removeImage,
//   setIntermediateImage,
// } from 'features/gallery/store/gallerySlice';

// import type { RootState } from 'app/store/store';
// import { addImageToStagingArea } from 'features/canvas/store/canvasSlice';
// import {
//   clearInitialImage,
//   initialImageSelected,
//   setInfillMethod,
//   // setInitialImage,
//   setMaskPath,
// } from 'features/parameters/store/generationSlice';
// import { tabMap } from 'features/ui/store/tabMap';
// import {
//   requestImages,
//   requestNewImages,
//   requestSystemConfig,
// } from './actions';

// /**
//  * Returns an object containing listener callbacks for socketio events.
//  * TODO: This file is large, but simple. Should it be split up further?
//  */
// const makeSocketIOListeners = (
//   store: MiddlewareAPI<Dispatch<AnyAction>, RootState>
// ) => {
//   const { dispatch, getState } = store;

//   return {
//     /**
//      * Callback to run when we receive a 'connect' event.
//      */
//     onConnect: () => {
//       try {
//         dispatch(setIsConnected(true));
//         dispatch(setCurrentStatus(i18n.t('common.statusConnected')));
//         dispatch(requestSystemConfig());
//         const gallery: GalleryState = getState().gallery;

//         if (gallery.categories.result.latest_mtime) {
//           dispatch(requestNewImages('result'));
//         } else {
//           dispatch(requestImages('result'));
//         }

//         if (gallery.categories.user.latest_mtime) {
//           dispatch(requestNewImages('user'));
//         } else {
//           dispatch(requestImages('user'));
//         }
//       } catch (e) {
//         console.error(e);
//       }
//     },
//     /**
//      * Callback to run when we receive a 'disconnect' event.
//      */
//     onDisconnect: () => {
//       try {
//         dispatch(setIsConnected(false));
//         dispatch(setCurrentStatus(i18n.t('common.statusDisconnected')));

//         dispatch(
//           addLogEntry({
//             timestamp: dateFormat(new Date(), 'isoDateTime'),
//             message: `Disconnected from server`,
//             level: 'warning',
//           })
//         );
//       } catch (e) {
//         console.error(e);
//       }
//     },
//     /**
//      * Callback to run when we receive a 'generationResult' event.
//      */
//     onGenerationResult: (data: InvokeAI.ImageResultResponse) => {
//       try {
//         const state = getState();
//         const { activeTab } = state.ui;
//         const { shouldLoopback } = state.postprocessing;
//         const { boundingBox: _, generationMode, ...rest } = data;

//         const newImage = {
//           uuid: uuidv4(),
//           ...rest,
//         };

//         if (['txt2img', 'img2img'].includes(generationMode)) {
//           dispatch(
//             addImage({
//               category: 'result',
//               image: { ...newImage, category: 'result' },
//             })
//           );
//         }

//         if (generationMode === 'unifiedCanvas' && data.boundingBox) {
//           const { boundingBox } = data;
//           dispatch(
//             addImageToStagingArea({
//               image: { ...newImage, category: 'temp' },
//               boundingBox,
//             })
//           );

//           if (state.canvas.shouldAutoSave) {
//             dispatch(
//               addImage({
//                 image: { ...newImage, category: 'result' },
//                 category: 'result',
//               })
//             );
//           }
//         }

//         // TODO: fix
//         // if (shouldLoopback) {
//         //   const activeTabName = tabMap[activeTab];
//         //   switch (activeTabName) {
//         //     case 'img2img': {
//         //       dispatch(initialImageSelected(newImage.uuid));
//         //       // dispatch(setInitialImage(newImage));
//         //       break;
//         //     }
//         //   }
//         // }

//         dispatch(clearIntermediateImage());

//         dispatch(
//           addLogEntry({
//             timestamp: dateFormat(new Date(), 'isoDateTime'),
//             message: `Image generated: ${data.url}`,
//           })
//         );
//       } catch (e) {
//         console.error(e);
//       }
//     },
//     /**
//      * Callback to run when we receive a 'intermediateResult' event.
//      */
//     onIntermediateResult: (data: InvokeAI.ImageResultResponse) => {
//       try {
//         dispatch(
//           setIntermediateImage({
//             uuid: uuidv4(),
//             ...data,
//             category: 'result',
//           })
//         );
//         if (!data.isBase64) {
//           dispatch(
//             addLogEntry({
//               timestamp: dateFormat(new Date(), 'isoDateTime'),
//               message: `Intermediate image generated: ${data.url}`,
//             })
//           );
//         }
//       } catch (e) {
//         console.error(e);
//       }
//     },
//     /**
//      * Callback to run when we receive an 'esrganResult' event.
//      */
//     onPostprocessingResult: (data: InvokeAI.ImageResultResponse) => {
//       try {
//         dispatch(
//           addImage({
//             category: 'result',
//             image: {
//               uuid: uuidv4(),
//               ...data,
//               category: 'result',
//             },
//           })
//         );

//         dispatch(
//           addLogEntry({
//             timestamp: dateFormat(new Date(), 'isoDateTime'),
//             message: `Postprocessed: ${data.url}`,
//           })
//         );
//       } catch (e) {
//         console.error(e);
//       }
//     },
//     /**
//      * Callback to run when we receive a 'progressUpdate' event.
//      * TODO: Add additional progress phases
//      */
//     onProgressUpdate: (data: InvokeAI.SystemStatus) => {
//       try {
//         dispatch(setIsProcessing(true));
//         dispatch(setSystemStatus(data));
//       } catch (e) {
//         console.error(e);
//       }
//     },
//     /**
//      * Callback to run when we receive a 'progressUpdate' event.
//      */
//     onError: (data: InvokeAI.ErrorResponse) => {
//       const { message, additionalData } = data;

//       if (additionalData) {
//         // TODO: handle more data than short message
//       }

//       try {
//         dispatch(
//           addLogEntry({
//             timestamp: dateFormat(new Date(), 'isoDateTime'),
//             message: `Server error: ${message}`,
//             level: 'error',
//           })
//         );
//         dispatch(errorOccurred());
//         dispatch(clearIntermediateImage());
//       } catch (e) {
//         console.error(e);
//       }
//     },
//     /**
//      * Callback to run when we receive a 'galleryImages' event.
//      */
//     onGalleryImages: (data: InvokeAI.GalleryImagesResponse) => {
//       const { images, areMoreImagesAvailable, category } = data;

//       /**
//        * the logic here ideally would be in the reducer but we have a side effect:
//        * generating a uuid. so the logic needs to be here, outside redux.
//        */

//       // Generate a UUID for each image
//       const preparedImages = images.map((image): InvokeAI._Image => {
//         return {
//           uuid: uuidv4(),
//           ...image,
//         };
//       });

//       dispatch(
//         addGalleryImages({
//           images: preparedImages,
//           areMoreImagesAvailable,
//           category,
//         })
//       );

//       dispatch(
//         addLogEntry({
//           timestamp: dateFormat(new Date(), 'isoDateTime'),
//           message: `Loaded ${images.length} images`,
//         })
//       );
//     },
//     /**
//      * Callback to run when we receive a 'processingCanceled' event.
//      */
//     onProcessingCanceled: () => {
//       dispatch(processingCanceled());

//       const { intermediateImage } = getState().gallery;

//       if (intermediateImage) {
//         if (!intermediateImage.isBase64) {
//           dispatch(
//             addImage({
//               category: 'result',
//               image: intermediateImage,
//             })
//           );
//           dispatch(
//             addLogEntry({
//               timestamp: dateFormat(new Date(), 'isoDateTime'),
//               message: `Intermediate image saved: ${intermediateImage.url}`,
//             })
//           );
//         }
//         dispatch(clearIntermediateImage());
//       }

//       dispatch(
//         addLogEntry({
//           timestamp: dateFormat(new Date(), 'isoDateTime'),
//           message: `Processing canceled`,
//           level: 'warning',
//         })
//       );
//     },
//     /**
//      * Callback to run when we receive a 'imageDeleted' event.
//      */
//     onImageDeleted: (data: InvokeAI.ImageDeletedResponse) => {
//       const { url } = data;

//       // remove image from gallery
//       dispatch(removeImage(data));

//       // remove references to image in options
//       const {
//         generation: { initialImage, maskPath },
//       } = getState();

//       if (
//         initialImage === url ||
//         (initialImage as InvokeAI._Image)?.url === url
//       ) {
//         dispatch(clearInitialImage());
//       }

//       if (maskPath === url) {
//         dispatch(setMaskPath(''));
//       }

//       dispatch(
//         addLogEntry({
//           timestamp: dateFormat(new Date(), 'isoDateTime'),
//           message: `Image deleted: ${url}`,
//         })
//       );
//     },
//     onSystemConfig: (data: InvokeAI.SystemConfig) => {
//       dispatch(setSystemConfig(data));
//       if (!data.infill_methods.includes('patchmatch')) {
//         dispatch(setInfillMethod(data.infill_methods[0]));
//       }
//     },
//     onFoundModels: (data: InvokeAI.FoundModelResponse) => {
//       const { search_folder, found_models } = data;
//       dispatch(setSearchFolder(search_folder));
//       dispatch(setFoundModels(found_models));
//     },
//     onNewModelAdded: (data: InvokeAI.ModelAddedResponse) => {
//       const { new_model_name, model_list, update } = data;
//       dispatch(setModelList(model_list));
//       dispatch(setIsProcessing(false));
//       dispatch(setCurrentStatus(i18n.t('modelManager.modelAdded')));
//       dispatch(
//         addLogEntry({
//           timestamp: dateFormat(new Date(), 'isoDateTime'),
//           message: `Model Added: ${new_model_name}`,
//           level: 'info',
//         })
//       );
//       dispatch(
//         addToast({
//           title: !update
//             ? `${i18n.t('modelManager.modelAdded')}: ${new_model_name}`
//             : `${i18n.t('modelManager.modelUpdated')}: ${new_model_name}`,
//           status: 'success',
//           duration: 2500,
//           isClosable: true,
//         })
//       );
//     },
//     onModelDeleted: (data: InvokeAI.ModelDeletedResponse) => {
//       const { deleted_model_name, model_list } = data;
//       dispatch(setModelList(model_list));
//       dispatch(setIsProcessing(false));
//       dispatch(
//         addLogEntry({
//           timestamp: dateFormat(new Date(), 'isoDateTime'),
//           message: `${i18n.t(
//             'modelManager.modelAdded'
//           )}: ${deleted_model_name}`,
//           level: 'info',
//         })
//       );
//       dispatch(
//         addToast({
//           title: `${i18n.t(
//             'modelManager.modelEntryDeleted'
//           )}: ${deleted_model_name}`,
//           status: 'success',
//           duration: 2500,
//           isClosable: true,
//         })
//       );
//     },
//     onModelConverted: (data: InvokeAI.ModelConvertedResponse) => {
//       const { converted_model_name, model_list } = data;
//       dispatch(setModelList(model_list));
//       dispatch(setCurrentStatus(i18n.t('common.statusModelConverted')));
//       dispatch(setIsProcessing(false));
//       dispatch(setIsCancelable(true));
//       dispatch(
//         addLogEntry({
//           timestamp: dateFormat(new Date(), 'isoDateTime'),
//           message: `Model converted: ${converted_model_name}`,
//           level: 'info',
//         })
//       );
//       dispatch(
//         addToast({
//           title: `${i18n.t(
//             'modelManager.modelConverted'
//           )}: ${converted_model_name}`,
//           status: 'success',
//           duration: 2500,
//           isClosable: true,
//         })
//       );
//     },
//     onModelsMerged: (data: InvokeAI.ModelsMergedResponse) => {
//       const { merged_models, merged_model_name, model_list } = data;
//       dispatch(setModelList(model_list));
//       dispatch(setCurrentStatus(i18n.t('common.statusMergedModels')));
//       dispatch(setIsProcessing(false));
//       dispatch(setIsCancelable(true));
//       dispatch(
//         addLogEntry({
//           timestamp: dateFormat(new Date(), 'isoDateTime'),
//           message: `Models merged: ${merged_models}`,
//           level: 'info',
//         })
//       );
//       dispatch(
//         addToast({
//           title: `${i18n.t('modelManager.modelsMerged')}: ${merged_model_name}`,
//           status: 'success',
//           duration: 2500,
//           isClosable: true,
//         })
//       );
//     },
//     onModelChanged: (data: InvokeAI.ModelChangeResponse) => {
//       const { model_name, model_list } = data;
//       dispatch(setModelList(model_list));
//       dispatch(setCurrentStatus(i18n.t('common.statusModelChanged')));
//       dispatch(setIsProcessing(false));
//       dispatch(setIsCancelable(true));
//       dispatch(
//         addLogEntry({
//           timestamp: dateFormat(new Date(), 'isoDateTime'),
//           message: `Model changed: ${model_name}`,
//           level: 'info',
//         })
//       );
//     },
//     onModelChangeFailed: (data: InvokeAI.ModelChangeResponse) => {
//       const { model_name, model_list } = data;
//       dispatch(setModelList(model_list));
//       dispatch(setIsProcessing(false));
//       dispatch(setIsCancelable(true));
//       dispatch(errorOccurred());
//       dispatch(
//         addLogEntry({
//           timestamp: dateFormat(new Date(), 'isoDateTime'),
//           message: `Model change failed: ${model_name}`,
//           level: 'error',
//         })
//       );
//     },
//     onTempFolderEmptied: () => {
//       dispatch(
//         addToast({
//           title: i18n.t('toast.tempFoldersEmptied'),
//           status: 'success',
//           duration: 2500,
//           isClosable: true,
//         })
//       );
//     },
//   };
// };

// export default makeSocketIOListeners;

export default {};
