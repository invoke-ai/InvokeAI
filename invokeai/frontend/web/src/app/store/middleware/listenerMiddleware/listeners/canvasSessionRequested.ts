import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import {
  layerAdded,
  layerImageAdded,
  sessionRequested,
  sessionStarted,
} from 'features/controlLayers/store/canvasV2Slice';
import { getImageDTO } from 'services/api/endpoints/images';
import { assert } from 'tsafe';

export const addCanvasSessionRequestedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: sessionRequested,
    effect: async (action, { getState, dispatch }) => {
      const initialImageObject = getState().canvasV2.initialImage.imageObject;
      if (initialImageObject) {
        // We have an initial image that needs to be converted to a layer
        dispatch(layerAdded());
        const newLayer = getState().canvasV2.layers.entities[0];
        assert(newLayer, 'Expected new layer to be created');
        const imageDTO = await getImageDTO(initialImageObject.image.name);
        assert(imageDTO, 'Unable to fetch initial image DTO');
        dispatch(layerImageAdded({ id: newLayer.id, imageDTO }));
      }

      dispatch(sessionStarted());
    },
  });
};
