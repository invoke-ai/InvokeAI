import { useAppDispatch } from 'app/store/storeHooks';
import {
  commitColorPickerColor,
  setColorPickerColor,
} from 'features/canvas/store/canvasSlice';
import {
  getCanvasBaseLayer,
  getCanvasStage,
} from 'features/canvas/util/konvaInstanceProvider';
import Konva from 'konva';
import { useCallback } from 'react';

const useColorPicker = () => {
  const dispatch = useAppDispatch();
  const canvasBaseLayer = getCanvasBaseLayer();
  const stage = getCanvasStage();

  const updateColorUnderCursor = useCallback(() => {
    if (!stage || !canvasBaseLayer) {
      return;
    }

    const position = stage.getPointerPosition();

    if (!position) {
      return;
    }

    const pixelRatio = Konva.pixelRatio;

    const [r, g, b, a] = canvasBaseLayer
      .getContext()
      .getImageData(
        position.x * pixelRatio,
        position.y * pixelRatio,
        1,
        1
      ).data;

    if (
      r === undefined ||
      g === undefined ||
      b === undefined ||
      a === undefined
    ) {
      return;
    }

    dispatch(setColorPickerColor({ r, g, b, a }));
  }, [canvasBaseLayer, dispatch, stage]);

  const commitColorUnderCursor = useCallback(() => {
    dispatch(commitColorPickerColor());
  }, [dispatch]);

  return { updateColorUnderCursor, commitColorUnderCursor };
};

export default useColorPicker;
