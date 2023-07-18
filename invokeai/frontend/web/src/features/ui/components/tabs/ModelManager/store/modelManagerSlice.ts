import { PayloadAction, createSlice } from '@reduxjs/toolkit';

type ModelManagerState = {
  searchFolder: string | null;
  advancedAddScanModel: string | null;
};

const initialModelManagerState: ModelManagerState = {
  searchFolder: null,
  advancedAddScanModel: null,
};

export const modelManagerSlice = createSlice({
  name: 'modelmanager',
  initialState: initialModelManagerState,
  reducers: {
    setSearchFolder: (state, action: PayloadAction<string | null>) => {
      state.searchFolder = action.payload;
    },
    setAdvancedAddScanModel: (state, action: PayloadAction<string | null>) => {
      state.advancedAddScanModel = action.payload;
    },
  },
});

export const { setSearchFolder, setAdvancedAddScanModel } =
  modelManagerSlice.actions;

export default modelManagerSlice.reducer;
