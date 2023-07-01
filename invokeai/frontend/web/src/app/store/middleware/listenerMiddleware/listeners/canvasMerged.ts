import { canvasMerged } from 'features/canvas/store/actions';
import { startAppListening } from '..';
import { log } from 'app/logging/useLogger';
import { addToast } from 'features/system/store/systemSlice';
import { imageUploaded } from 'services/api/thunks/image';
import { setMergedCanvas } from 'features/canvas/store/canvasSlice';
import { getCanvasBaseLayer } from 'features/canvas/util/konvaInstanceProvider';
import { getFullBaseLayerBlob } from 'features/canvas/util/getFullBaseLayerBlob';

const moduleLog = log.child({ namespace: 'canvasCopiedToClipboardListener' });

export const addCanvasMergedListener = () => {
  startAppListening({
    actionCreator: canvasMerged,
    effect: async (action, { dispatch, getState, take }) => {
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

      const imageUploadedRequest = dispatch(
        imageUploaded({
          file: new File([blob], 'mergedCanvas.png', {
            type: 'image/png',
          }),
          image_category: 'general',
          is_intermediate: true,
          postUploadAction: {
            type: 'TOAST_CANVAS_MERGED',
          },
        })
      );

      const [{ payload }] = await take(
        (
          uploadedImageAction
        ): uploadedImageAction is ReturnType<typeof imageUploaded.fulfilled> =>
          imageUploaded.fulfilled.match(uploadedImageAction) &&
          uploadedImageAction.meta.requestId === imageUploadedRequest.requestId
      );

      const { image_name } = payload;

      dispatch(
        setMergedCanvas({
          kind: 'image',
          layer: 'base',
          imageName: image_name,
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
