import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { imageMetadataReceived } from 'services/thunks/image';
import {
  ResultsImageDTO,
  resultsAdapter,
} from 'features/gallery/store/resultsSlice';
import {
  UploadsImageDTO,
  uploadsAdapter,
} from 'features/gallery/store/uploadsSlice';

const moduleLog = log.child({ namespace: 'image' });

export const addImageMetadataReceivedFulfilledListener = () => {
  startAppListening({
    actionCreator: imageMetadataReceived.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const image = action.payload;
      moduleLog.debug({ data: { image } }, 'Image metadata received');

      if (image.image_type === 'results') {
        resultsAdapter.upsertOne(
          getState().results,
          action.payload as ResultsImageDTO
        );
      }

      if (image.image_type === 'uploads') {
        uploadsAdapter.upsertOne(
          getState().uploads,
          action.payload as UploadsImageDTO
        );
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
