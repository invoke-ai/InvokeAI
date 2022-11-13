import { Spinner } from '@chakra-ui/react';
import { useLayoutEffect, useRef } from 'react';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import {
  baseCanvasImageSelector,
  CanvasState,
  setStageDimensions,
  setStageScale,
} from 'features/canvas/canvasSlice';
import { createSelector } from '@reduxjs/toolkit';

const canvasResizerSelector = createSelector(
  (state: RootState) => state.canvas,
  baseCanvasImageSelector,
  activeTabNameSelector,
  (canvas: CanvasState, baseCanvasImage, activeTabName) => {
    const { doesCanvasNeedScaling } = canvas;

    return {
      doesCanvasNeedScaling,
      activeTabName,
      baseCanvasImage,
    };
  }
);

const IAICanvasResizer = () => {
  const dispatch = useAppDispatch();
  const { doesCanvasNeedScaling, activeTabName, baseCanvasImage } =
    useAppSelector(canvasResizerSelector);

  const ref = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    window.setTimeout(() => {
      if (!ref.current) return;
      const { width: imageWidth, height: imageHeight } = baseCanvasImage?.image
        ? baseCanvasImage.image
        : { width: 512, height: 512 };
      const { clientWidth, clientHeight } = ref.current;

      const scale = Math.min(
        1,
        Math.min(clientWidth / imageWidth, clientHeight / imageHeight)
      );

      dispatch(setStageScale(scale));

      if (activeTabName === 'inpainting') {
        dispatch(
          setStageDimensions({
            width: Math.floor(imageWidth * scale),
            height: Math.floor(imageHeight * scale),
          })
        );
      } else if (activeTabName === 'outpainting') {
        dispatch(
          setStageDimensions({
            width: Math.floor(clientWidth),
            height: Math.floor(clientHeight),
          })
        );
      }
    }, 0);
  }, [dispatch, baseCanvasImage, doesCanvasNeedScaling, activeTabName]);

  return (
    <div ref={ref} className="inpainting-canvas-area">
      <Spinner thickness="2px" speed="1s" size="xl" />
    </div>
  );
};

export default IAICanvasResizer;
