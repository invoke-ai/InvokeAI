import { PayloadAction, createSlice } from '@reduxjs/toolkit';

type ModelManagerState = {
  searchFolder: string | null;
};

const initialModelManagerState: ModelManagerState = {
  searchFolder: null,
};

export const modelManagerSlice = createSlice({
  name: 'modelmanager',
  initialState: initialModelManagerState,
  reducers: {
    setSearchFolder: (state, action: PayloadAction<string | null>) => {
      state.searchFolder = action.payload;
    },
  },
});

export const { setSearchFolder } = modelManagerSlice.actions;

export default modelManagerSlice.reducer;
