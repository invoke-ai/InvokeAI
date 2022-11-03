import { Spinner } from '@chakra-ui/react';
import { useLayoutEffect, useRef } from 'react';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import { setStageDimensions, setStageScale } from './inpaintingSlice';

const InpaintingCanvasPlaceholder = () => {
  const dispatch = useAppDispatch();
  const { doesCanvasNeedScaling, imageToInpaint } = useAppSelector(
    (state: RootState) => state.inpainting
  );
  const ref = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    window.setTimeout(() => {
      if (!ref.current || !imageToInpaint) return;

      const width = ref.current.clientWidth;
      const height = ref.current.clientHeight;

      dispatch(setStageDimensions({ width, height }));

      // dispatch(setStageScale(1));
      const scale = Math.min(
        1,
        Math.min(width / imageToInpaint.width, height / imageToInpaint.height)
      );

      dispatch(setStageScale(scale));
    }, 0);
  }, [dispatch, imageToInpaint, doesCanvasNeedScaling]);

  return (
    <div ref={ref} className="inpainting-canvas-area">
      <Spinner thickness="2px" speed="1s" size="xl" />
    </div>
  );
};

export default InpaintingCanvasPlaceholder;
