import { setUploadProgress } from 'features/system/store/systemSlice';
import { socketUploadProgress } from 'services/events/actions';

import { startAppListening } from '../..';

export const addSocketUploadProgressEventListener = () => {
  startAppListening({
    actionCreator: socketUploadProgress,
    effect: (action, { dispatch }) => {
    //   console.log(`this is the event listener ${action.payload}`); // TODO: remove, debugging purposes
      const { progress, processed, total } = action.payload.data;
      dispatch(setUploadProgress({ progress, processed, total }));

      if (progress === 100) {
        setTimeout(() => {
          dispatch(setUploadProgress(null));
        }, 2500);
      }
    },
  });
};
