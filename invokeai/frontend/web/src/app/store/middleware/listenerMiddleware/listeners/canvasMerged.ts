import { canvasMerged } from 'features/canvas/store/actions';
import { startAppListening } from '..';
import { log } from 'app/logging/useLogger';
import { getBaseLayerBlob } from 'features/canvas/util/getBaseLayerBlob';
import { addToast } from 'features/system/store/systemSlice';
import { imageUploaded } from 'services/thunks/image';
import { v4 as uuidv4 } from 'uuid';
import { setMergedCanvas } from 'features/canvas/store/canvasSlice';
import { getCanvasBaseLayer } from 'features/canvas/util/konvaInstanceProvider';

const moduleLog = log.child({ namespace: 'canvasCopiedToClipboardListener' });

export const addCanvasMergedListener = () => {
  startAppListening({
    actionCreator: canvasMerged,
    effect: async (action, { dispatch, getState, take }) => {
      const state = getState();

      const blob = await getBaseLayerBlob(state, true);

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

      const filename = `mergedCanvas_${uuidv4()}.png`;

      dispatch(
        imageUploaded({
          formData: {
            file: new File([blob], filename, { type: 'image/png' }),
          },
          imageCategory: 'general',
          isIntermediate: true,
        })
      );

      const [{ payload }] = await take(
        (action): action is ReturnType<typeof imageUploaded.fulfilled> =>
          imageUploaded.fulfilled.match(action) &&
          action.meta.arg.formData.file.name === filename
      );

      const mergedCanvasImage = payload;

      dispatch(
        setMergedCanvas({
          kind: 'image',
          layer: 'base',
          image: mergedCanvasImage,
          ...baseLayerRect,
        })
      );

      dispatch(
        addToast({
          title: 'Canvas Merged',
          status: 'success',
        })
      );
    },
  });
};
