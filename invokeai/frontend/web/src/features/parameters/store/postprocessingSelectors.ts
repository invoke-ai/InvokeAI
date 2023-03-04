import { RootState } from 'app/store';

export const postprocessingSelector = (state: RootState) =>
  state.postprocessing;
