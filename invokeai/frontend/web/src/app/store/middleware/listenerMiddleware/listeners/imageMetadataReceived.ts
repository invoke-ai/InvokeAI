import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { imageMetadataReceived } from 'services/thunks/image';
import {
  ResultsImageDTO,
  resultUpserted,
} from 'features/gallery/store/resultsSlice';
import {
  UploadsImageDTO,
  uploadUpserted,
} from 'features/gallery/store/uploadsSlice';

const moduleLog = log.child({ namespace: 'image' });

export const addImageMetadataReceivedFulfilledListener = () => {
  startAppListening({
    actionCreator: imageMetadataReceived.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const image = action.payload;
      moduleLog.debug({ data: { image } }, 'Image metadata received');

      if (image.image_type === 'results') {
        console.log('upsert results');
        dispatch(resultUpserted(action.payload as ResultsImageDTO));
      }

      if (image.image_type === 'uploads') {
        console.log('upsert uploads');
        dispatch(uploadUpserted(action.payload as UploadsImageDTO));
      }
    },
  });
};

export const addImageMetadataReceivedRejectedListener = () => {
  startAppListening({
    actionCreator: imageMetadataReceived.rejected,
    effect: (action, { getState, dispatch }) => {
      moduleLog.debug(
        { data: { image: action.meta.arg } },
        'Problem receiving image metadata'
      );
    },
  });
};
