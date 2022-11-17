import { Spinner } from '@chakra-ui/react';
import { useLayoutEffect, useRef } from 'react';
import { useAppDispatch, useAppSelector } from 'app/store';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import {
  resizeAndScaleCanvas,
  resizeCanvas,
  setCanvasContainerDimensions,
  setDoesCanvasNeedScaling,
} from 'features/canvas/store/canvasSlice';
import { createSelector } from '@reduxjs/toolkit';
import { canvasSelector, initialCanvasImageSelector } from 'features/canvas/store/canvasSelectors';

const canvasResizerSelector = createSelector(
  canvasSelector,
  initialCanvasImageSelector,
  activeTabNameSelector,
  (canvas, initialCanvasImage, activeTabName) => {
    const {
      doesCanvasNeedScaling,
      shouldLockToInitialImage,
      isCanvasInitialized,
    } = canvas;
    return {
      doesCanvasNeedScaling,
      shouldLockToInitialImage,
      activeTabName,
      initialCanvasImage,
      isCanvasInitialized,
    };
  }
);

const IAICanvasResizer = () => {
  const dispatch = useAppDispatch();
  const {
    doesCanvasNeedScaling,
    shouldLockToInitialImage,
    activeTabName,
    initialCanvasImage,
    isCanvasInitialized,
  } = useAppSelector(canvasResizerSelector);

  const ref = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    window.setTimeout(() => {
      if (!ref.current) return;

      const { clientWidth, clientHeight } = ref.current;

      if (!initialCanvasImage?.image) return;

      dispatch(
        setCanvasContainerDimensions({
          width: clientWidth,
          height: clientHeight,
        })
      );

      if (!isCanvasInitialized || shouldLockToInitialImage) {
        dispatch(resizeAndScaleCanvas());
      } else {
        dispatch(resizeCanvas());
      }

      dispatch(setDoesCanvasNeedScaling(false));
    }, 0);
  }, [
    dispatch,
    initialCanvasImage,
    doesCanvasNeedScaling,
    activeTabName,
    isCanvasInitialized,
    shouldLockToInitialImage,
  ]);

  return (
    <div ref={ref} className="inpainting-canvas-area">
      <Spinner thickness="2px" speed="1s" size="xl" />
    </div>
  );
};

export default IAICanvasResizer;
