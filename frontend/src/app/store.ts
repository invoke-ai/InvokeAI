import { configureStore } from '@reduxjs/toolkit';
import sdReducer from '../features/sd/sdSlice';
import galleryReducer from '../features/gallery/gallerySlice';
import systemReducer from '../features/system/systemSlice';

/*
Store Slices
- sd: image generation parameters
- gallery: image gallery
- system: logs, site settings, etc
*/

export const store = configureStore({
  reducer: {
    sd: sdReducer,
    gallery: galleryReducer,
    system: systemReducer,
  },
  // devTools: {
  //   trace: true,
  // },
});

// Infer the `RootState` and `AppDispatch` types from the store itself
export type RootState = ReturnType<typeof store.getState>;
// Inferred type: {posts: PostsState, comments: CommentsState, users: UsersState}
export type AppDispatch = typeof store.dispatch;
