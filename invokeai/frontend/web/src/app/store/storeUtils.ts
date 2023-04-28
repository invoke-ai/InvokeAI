import { createAsyncThunk } from '@reduxjs/toolkit';
import { AppDispatch, RootState } from 'app/store/store';

// https://redux-toolkit.js.org/usage/usage-with-typescript#defining-a-pre-typed-createasyncthunk
export const createAppAsyncThunk = createAsyncThunk.withTypes<{
  state: RootState;
  dispatch: AppDispatch;
}>();
