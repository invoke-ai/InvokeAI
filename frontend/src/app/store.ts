import { configureStore } from '@reduxjs/toolkit';
import sdReducer from './sdSlice';

export const store = configureStore({
  reducer: {
    sd: sdReducer,
  },
  // devTools: {
  //   trace: true,
  // },
});

// Infer the `RootState` and `AppDispatch` types from the store itself
export type RootState = ReturnType<typeof store.getState>;
// Inferred type: {posts: PostsState, comments: CommentsState, users: UsersState}
export type AppDispatch = typeof store.dispatch;
