import { Flex, Spinner } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  canvasSelector,
  initialCanvasImageSelector,
} from 'features/canvas/store/canvasSelectors';
import {
  resizeAndScaleCanvas,
  resizeCanvas,
  setCanvasContainerDimensions,
  setDoesCanvasNeedScaling,
} from 'features/canvas/store/canvasSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useLayoutEffect, useRef } from 'react';

const canvasResizerSelector = createSelector(
  canvasSelector,
  initialCanvasImageSelector,
  activeTabNameSelector,
  (canvas, initialCanvasImage, activeTabName) => {
    const { doesCanvasNeedScaling, isCanvasInitialized } = canvas;
    return {
      doesCanvasNeedScaling,
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
    activeTabName,
    initialCanvasImage,
    isCanvasInitialized,
  } = useAppSelector(canvasResizerSelector);

  const ref = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    window.setTimeout(() => {
      if (!ref.current) return;

      const { clientWidth, clientHeight } = ref.current;

      dispatch(
        setCanvasContainerDimensions({
          width: clientWidth,
          height: clientHeight,
        })
      );

      if (!isCanvasInitialized) {
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
  ]);

  return (
    <Flex
      ref={ref}
      sx={{
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 4,
        width: '100%',
        height: '100%',
      }}
    >
      <Spinner thickness="2px" size="xl" />
    </Flex>
  );
};

export default IAICanvasResizer;
