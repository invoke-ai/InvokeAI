import { useAppDispatch } from 'app/store/storeHooks';
import Konva from 'konva';
import {
  commitColorPickerColor,
  setColorPickerColor,
} from '../store/canvasSlice';
import {
  getCanvasBaseLayer,
  getCanvasStage,
} from '../util/konvaInstanceProvider';

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

      if (!(a && r && g && b)) {
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
