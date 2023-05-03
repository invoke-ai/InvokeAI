// import { createAction } from '@reduxjs/toolkit';
// import * as InvokeAI from 'app/types/invokeai';
// import { GalleryCategory } from 'features/gallery/store/gallerySlice';
// import { InvokeTabName } from 'features/ui/store/tabMap';

// /**
//  * We can't use redux-toolkit's createSlice() to make these actions,
//  * because they have no associated reducer. They only exist to dispatch
//  * requests to the server via socketio. These actions will be handled
//  * by the middleware.
//  */

// export const generateImage = createAction<InvokeTabName>(
//   'socketio/generateImage'
// );
// export const runESRGAN = createAction<InvokeAI._Image>('socketio/runESRGAN');
// export const runFacetool = createAction<InvokeAI._Image>(
//   'socketio/runFacetool'
// );
// export const deleteImage = createAction<InvokeAI._Image>(
//   'socketio/deleteImage'
// );
// export const requestImages = createAction<GalleryCategory>(
//   'socketio/requestImages'
// );
// export const requestNewImages = createAction<GalleryCategory>(
//   'socketio/requestNewImages'
// );
// export const cancelProcessing = createAction<undefined>(
//   'socketio/cancelProcessing'
// );

// export const requestSystemConfig = createAction<undefined>(
//   'socketio/requestSystemConfig'
// );

// export const searchForModels = createAction<string>('socketio/searchForModels');

// export const addNewModel = createAction<
//   InvokeAI.InvokeModelConfigProps | InvokeAI.InvokeDiffusersModelConfigProps
// >('socketio/addNewModel');

// export const deleteModel = createAction<string>('socketio/deleteModel');

// export const convertToDiffusers =
//   createAction<InvokeAI.InvokeModelConversionProps>(
//     'socketio/convertToDiffusers'
//   );

// export const mergeDiffusersModels =
//   createAction<InvokeAI.InvokeModelMergingProps>(
//     'socketio/mergeDiffusersModels'
//   );

// export const requestModelChange = createAction<string>(
//   'socketio/requestModelChange'
// );

// export const saveStagingAreaImageToGallery = createAction<string>(
//   'socketio/saveStagingAreaImageToGallery'
// );

// export const emptyTempFolder = createAction<undefined>(
//   'socketio/requestEmptyTempFolder'
// );

export default {};
