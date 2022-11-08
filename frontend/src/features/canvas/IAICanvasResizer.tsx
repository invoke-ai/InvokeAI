import { Spinner } from '@chakra-ui/react';
import { useLayoutEffect, useRef } from 'react';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import {
  baseCanvasImageSelector,
  CanvasState,
  currentCanvasSelector,
  GenericCanvasState,
  setStageDimensions,
  setStageScale,
} from 'features/canvas/canvasSlice';
import { createSelector } from '@reduxjs/toolkit';
import * as InvokeAI from 'app/invokeai';
import { first } from 'lodash';

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
      if (!ref.current || !baseCanvasImage) return;

      const width = ref.current.clientWidth;
      const height = ref.current.clientHeight;

      const scale = Math.min(
        1,
        Math.min(width / baseCanvasImage.width, height / baseCanvasImage.height)
      );

      dispatch(setStageScale(scale));

      if (activeTabName === 'inpainting') {
        dispatch(
          setStageDimensions({
            width: Math.floor(baseCanvasImage.width * scale),
            height: Math.floor(baseCanvasImage.height * scale),
          })
        );
      } else if (activeTabName === 'outpainting') {
        dispatch(
          setStageDimensions({
            width: Math.floor(width),
            height: Math.floor(height),
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
