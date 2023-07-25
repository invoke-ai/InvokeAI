import { $logger } from 'app/logging/logger';
import { canvasMerged } from 'features/canvas/store/actions';
import { setMergedCanvas } from 'features/canvas/store/canvasSlice';
import { getFullBaseLayerBlob } from 'features/canvas/util/getFullBaseLayerBlob';
import { getCanvasBaseLayer } from 'features/canvas/util/konvaInstanceProvider';
import { addToast } from 'features/system/store/systemSlice';
import { imagesApi } from 'services/api/endpoints/images';
import { startAppListening } from '..';

export const addCanvasMergedListener = () => {
  startAppListening({
    actionCreator: canvasMerged,
    effect: async (action, { dispatch }) => {
      const moduleLog = $logger
        .get()
        .child({ namespace: 'canvasCopiedToClipboardListener' });
      const blob = await getFullBaseLayerBlob();

      if (!blob) {
        moduleLog.error('Problem getting base layer blob');
        dispatch(
          addToast({
            title: 'Problem Merging Canvas',
            description: 'Unable to export base layer',
            status: 'error',
          })
        );
        return;
      }

      const canvasBaseLayer = getCanvasBaseLayer();

      if (!canvasBaseLayer) {
        moduleLog.error('Problem getting canvas base layer');
        dispatch(
          addToast({
            title: 'Problem Merging Canvas',
            description: 'Unable to export base layer',
            status: 'error',
          })
        );
        return;
      }

      const baseLayerRect = canvasBaseLayer.getClientRect({
        relativeTo: canvasBaseLayer.getParent(),
      });

      const imageDTO = await dispatch(
        imagesApi.endpoints.uploadImage.initiate({
          file: new File([blob], 'mergedCanvas.png', {
            type: 'image/png',
          }),
          image_category: 'general',
          is_intermediate: true,
          postUploadAction: {
            type: 'TOAST',
            toastOptions: { title: 'Canvas Merged' },
          },
        })
      ).unwrap();

      // TODO: I can't figure out how to do the type narrowing in the `take()` so just brute forcing it here
      const { image_name } = imageDTO;

      dispatch(
        setMergedCanvas({
          kind: 'image',
          layer: 'base',
          imageName: image_name,
          ...baseLayerRect,
        })
      );
    },
  });
};
