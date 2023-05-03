import { RootState } from 'app/store/store';

export const postprocessingSelector = (state: RootState) =>
  state.postprocessing;
