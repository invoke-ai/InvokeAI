import { log } from 'app/logging/useLogger';
import { serializeError } from 'serialize-error';
import { imagesApi } from 'services/api/endpoints/images';
import { imagesLoaded } from 'services/api/thunks/image';
import { startAppListening } from '..';

const moduleLog = log.child({ namespace: 'gallery' });

export const addImagesLoadedListener = () => {
  startAppListening({
    actionCreator: imagesLoaded.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const { items } = action.payload;
      moduleLog.debug(
        { data: { payload: action.payload } },
        `Loaded ${items.length} images`
      );

      items.forEach((image) => {
        dispatch(
          imagesApi.util.upsertQueryData('getImageDTO', image.image_name, image)
        );
      });
    },
  });

  startAppListening({
    actionCreator: imagesLoaded.rejected,
    effect: (action, { getState, dispatch }) => {
      if (action.payload) {
        moduleLog.debug(
          { data: { error: serializeError(action.payload) } },
          'Problem loading images'
        );
      }
    },
  });
};
