import { createAction } from '@reduxjs/toolkit';
import { GalleryCategory } from '../../features/gallery/gallerySlice';
import { InvokeTabName } from '../../features/tabs/InvokeTabs';
import * as InvokeAI from '../invokeai';


/**
 * We can't use redux-toolkit's createSlice() to make these actions,
 * because they have no associated reducer. They only exist to dispatch
 * requests to the server via socketio. These actions will be handled
 * by the middleware.
 */

export const generateImage = createAction<InvokeTabName>(
  'socketio/generateImage'
);
export const runESRGAN = createAction<InvokeAI.Image>('socketio/runESRGAN');
export const runFacetool = createAction<InvokeAI.Image>('socketio/runFacetool');
export const deleteImage = createAction<InvokeAI.Image>('socketio/deleteImage');
export const requestImages = createAction<GalleryCategory>(
  'socketio/requestImages'
);
export const requestNewImages = createAction<GalleryCategory>(
  'socketio/requestNewImages'
);
export const cancelProcessing = createAction<undefined>(
  'socketio/cancelProcessing'
);
export const uploadImage = createAction<InvokeAI.UploadImagePayload>('socketio/uploadImage');
export const uploadMaskImage = createAction<File>('socketio/uploadMaskImage');

export const requestSystemConfig = createAction<undefined>(
  'socketio/requestSystemConfig'
);

export const requestModelChange = createAction<string>(
  'socketio/requestModelChange'
);
