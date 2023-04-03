import { isFulfilled, Middleware, MiddlewareAPI } from '@reduxjs/toolkit';
import { v4 as uuidv4 } from 'uuid';

import { emitSubscribe } from 'app/nodesSocketio/actions';
import { AppDispatch, RootState } from 'app/store';
import { setSessionId } from './apiSlice';
import { uploadImage } from './thunks/image';
import { createSession, invokeSession } from './thunks/session';
import * as InvokeAI from 'app/invokeai';
import { addImage } from 'features/gallery/store/gallerySlice';
import { tabMap } from 'features/ui/store/tabMap';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import { setInitialImage } from 'features/parameters/store/generationSlice';

/**
 * `redux-toolkit` provides nice matching utilities, which can be used as type guards
 * See: https://redux-toolkit.js.org/api/matching-utilities
 */

const isFulfilledCreateSession = isFulfilled(createSession);
const isFulfilledUploadImage = isFulfilled(uploadImage);

export const invokeMiddleware: Middleware =
  (store: MiddlewareAPI<AppDispatch, RootState>) => (next) => (action) => {
    const { dispatch, getState } = store;

    if (isFulfilledCreateSession(action)) {
      const sessionId = action.payload.id;
      console.log('createSession.fulfilled');

      dispatch(setSessionId(sessionId));
      dispatch(emitSubscribe(sessionId));
      dispatch(invokeSession({ sessionId }));
    } else if (isFulfilledUploadImage(action)) {
      const uploadLocation = action.payload;
      console.log('uploadImage.fulfilled');

      const newImage: InvokeAI.Image = {
        uuid: uuidv4(),
        category: 'user',
        url: uploadLocation,
        width: 512,
        height: 512,
        mtime: new Date().getTime(),
        thumbnail: uploadLocation,
      };

      dispatch(addImage({ image: newImage, category: 'user' }));

      const { activeTab } = getState().ui;
      const activeTabName = tabMap[activeTab];
      if (activeTabName === 'unifiedCanvas') {
        dispatch(setInitialCanvasImage(newImage));
      } else if (activeTabName === 'img2img') {
        dispatch(setInitialImage(newImage));
      }
    } else {
      next(action);
    }
  };
