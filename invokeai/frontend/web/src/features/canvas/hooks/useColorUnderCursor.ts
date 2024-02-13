import { useAppDispatch } from 'app/store/storeHooks';
import { $canvasBaseLayer, $canvasStage, $tool } from 'features/canvas/store/canvasNanostore';
import { commitColorPickerColor, setColorPickerColor } from 'features/canvas/store/canvasSlice';
import Konva from 'konva';
import { useCallback } from 'react';

const useColorPicker = () => {
  const dispatch = useAppDispatch();

  const updateColorUnderCursor = useCallback(() => {
    const stage = $canvasStage.get();
    const canvasBaseLayer = $canvasBaseLayer.get();
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
      .getImageData(position.x * pixelRatio, position.y * pixelRatio, 1, 1).data;

    if (r === undefined || g === undefined || b === undefined || a === undefined) {
      return;
    }

    dispatch(setColorPickerColor({ r, g, b, a }));
  }, [dispatch]);

  const commitColorUnderCursor = useCallback(() => {
    dispatch(commitColorPickerColor());
    $tool.set('brush');
  }, [dispatch]);

  return { updateColorUnderCursor, commitColorUnderCursor };
};

export default useColorPicker;
