import {
  combineReducers,
  configureStore,
} from '@reduxjs/toolkit';
import { persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage'; // defaults to localStorage for web

import sdReducer from '../features/sd/sdSlice';
import galleryReducer from '../features/gallery/gallerySlice';
import systemReducer from '../features/system/systemSlice';

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

export const store = configureStore({
  reducer: persistedReducer,
  // devTools: {
  //   trace: true,
  // },
});

// Infer the `RootState` and `AppDispatch` types from the store itself
export type RootState = ReturnType<typeof store.getState>;
// Inferred type: {posts: PostsState, comments: CommentsState, users: UsersState}
export type AppDispatch = typeof store.dispatch;
