import { combineReducers, configureStore } from '@reduxjs/toolkit';
import { persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage'; // defaults to localStorage for web

import sdReducer from '../features/sd/sdSlice';
import galleryReducer from '../features/gallery/gallerySlice';
import systemReducer from '../features/system/systemSlice';
import { socketioMiddleware } from './socketio';

const reducers = combineReducers({
  sd: sdReducer,
  gallery: galleryReducer,
  system: systemReducer,
});

const persistConfig = {
  key: 'root',
  storage,
};

const persistedReducer = persistReducer(persistConfig, reducers);

/*
  The frontend needs to be distributed as a production build, so
  we cannot reasonably ask users to edit the JS and specify the
  host and port on which the socket.io server will run.

  The solution is to allow server script to be run with arguments
  (or just edited) providing the host and port. Then, the server
  serves a route `/socketio_config` which responds with the host
  and port.

  When the frontend loads, it synchronously requests that route
  and thus gets the host and port. This requires a suspicious
  fetch somewhere, and the store setup seems like as good a place
  as any to make this fetch request.
*/

interface SocketIOConfig {
  host: string;
  port: number;
}

const fetchSocketioConfig = async (): Promise<SocketIOConfig> => {
  const uri = 'socketio_config';
  const response = await fetch(uri);

  if (response.status === 200) {
    const config = await response.json();
    return config;
  }

  console.error(
    `Unable to get server config - will use default host (localhost) and port (9090)`,
    response
  );

  return { host: 'localhost', port: 9090 };
};

const socketioConfig = await fetchSocketioConfig();

// Continue with store setup
export const store = configureStore({
  reducer: persistedReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      // redux-persist sometimes needs to have a function in redux, need to disable this check
      serializableCheck: false,
    }).concat(socketioMiddleware(socketioConfig)),
});

// Infer the `RootState` and `AppDispatch` types from the store itself
export type RootState = ReturnType<typeof store.getState>;
// Inferred type: {posts: PostsState, comments: CommentsState, users: UsersState}
export type AppDispatch = typeof store.dispatch;
