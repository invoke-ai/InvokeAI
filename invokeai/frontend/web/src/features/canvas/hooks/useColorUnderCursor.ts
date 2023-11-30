import { useAppDispatch } from 'app/store/storeHooks';
import Konva from 'konva';
import {
  commitColorPickerColor,
  setColorPickerColor,
} from 'features/canvas/store/canvasSlice';
import {
  getCanvasBaseLayer,
  getCanvasStage,
} from 'features/canvas/util/konvaInstanceProvider';

const useColorPicker = () => {
  const dispatch = useAppDispatch();
  const canvasBaseLayer = getCanvasBaseLayer();
  const stage = getCanvasStage();

  return {
    updateColorUnderCursor: () => {
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
    },
    commitColorUnderCursor: () => {
      dispatch(commitColorPickerColor());
    },
  };
};

export default useColorPicker;
