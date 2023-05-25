import { invocationComplete } from 'services/events/actions';
import { isImageOutput } from 'services/types/guards';
import { imageMetadataReceived } from 'services/thunks/image';
import { startAppListening } from '..';
import { progressImageSet } from '../../../../../features/system/store/systemSlice';

export const addImageMetadataReceivedListener = () => {
  startAppListening({
    predicate: (action) => {
      if (
        invocationComplete.match(action) &&
        isImageOutput(action.payload.data.result)
      ) {
        return true;
      }
      return false;
    },
    effect: async (action, { getState, dispatch, take }) => {
      if (imageMetadataReceived.fulfilled.match(action)) {
        return;
      }

      dispatch(progressImageSet(null));
    },
  });
};
