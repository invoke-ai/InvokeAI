import { Spinner } from '@chakra-ui/react';
import { useLayoutEffect, useRef } from 'react';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import { setStageScale } from './inpaintingSlice';

const InpaintingCanvasPlaceholder = () => {
  const dispatch = useAppDispatch();
  const { needsRepaint, imageToInpaint } = useAppSelector(
    (state: RootState) => state.inpainting
  );
  const ref = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    if (!ref.current || !imageToInpaint) return;

    const width = ref.current.clientWidth;
    const height = ref.current.clientHeight;

    const scale = Math.min(
      1,
      Math.min(width / imageToInpaint.width, height / imageToInpaint.height)
    );

    dispatch(setStageScale(scale));
  }, [dispatch, imageToInpaint, needsRepaint]);

  return (
    <div ref={ref} className="inpainting-canvas-container">
      <Spinner thickness="2px" speed="1s" size="xl" />
    </div>
  );
};

export default InpaintingCanvasPlaceholder;
