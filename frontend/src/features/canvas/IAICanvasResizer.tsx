import { Spinner } from '@chakra-ui/react';
import { useLayoutEffect, useRef } from 'react';
import { useAppDispatch, useAppSelector } from 'app/store';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import {
  initialCanvasImageSelector,
  canvasSelector,
  resizeAndScaleCanvas,
  resizeCanvas,
  setCanvasContainerDimensions,
  setDoesCanvasNeedScaling,
} from 'features/canvas/canvasSlice';
import { createSelector } from '@reduxjs/toolkit';

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

      const { width: imageWidth, height: imageHeight } =
        initialCanvasImage.image;

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
      // }
      // if ((activeTabName === 'inpainting') && initialCanvasImage?.image) {
      //   const { width: imageWidth, height: imageHeight } =
      //     initialCanvasImage.image;

      //   const scale = Math.min(
      //     1,
      //     Math.min(clientWidth / imageWidth, clientHeight / imageHeight)
      //   );

      //   dispatch(setStageScale(scale));

      //   dispatch(
      //     setStageDimensions({
      //       width: Math.floor(imageWidth * scale),
      //       height: Math.floor(imageHeight * scale),
      //     })
      //   );
      //   dispatch(setDoesCanvasNeedScaling(false));
      // } else if (activeTabName === 'outpainting') {
      //   dispatch(
      //     setStageDimensions({
      //       width: Math.floor(clientWidth),
      //       height: Math.floor(clientHeight),
      //     })
      //   );
      //   dispatch(setDoesCanvasNeedScaling(false));
      // }
    }, 0);
  }, [
    dispatch,
    initialCanvasImage,
    doesCanvasNeedScaling,
    activeTabName,
    isCanvasInitialized,
  ]);

  return (
    <div ref={ref} className="inpainting-canvas-area">
      <Spinner thickness="2px" speed="1s" size="xl" />
    </div>
  );
};

export default IAICanvasResizer;
