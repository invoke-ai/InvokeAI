import { createSelector } from '@reduxjs/toolkit';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import Konva from 'konva';
import { KonvaEventObject } from 'konva/lib/Node';
import _ from 'lodash';
import { MutableRefObject, useCallback } from 'react';
import {
  baseCanvasImageSelector,
  currentCanvasSelector,
  GenericCanvasState,
  setStageCoordinates,
  setStageScale,
  shouldLockToInitialImageSelector,
} from '../canvasSlice';
import {
  CANVAS_SCALE_BY,
  MAX_CANVAS_SCALE,
  MIN_CANVAS_SCALE,
} from '../util/constants';

const selector = createSelector(
  [
    (state: RootState) => state.canvas,
    activeTabNameSelector,
    currentCanvasSelector,
    baseCanvasImageSelector,
    shouldLockToInitialImageSelector,
  ],
  (
    canvas,
    activeTabName,
    currentCanvas,
    baseCanvasImage,
    shouldLockToInitialImage
  ) => {
    const {
      isMoveStageKeyHeld,
      stageScale,
      stageDimensions,
      minimumStageScale,
    } = currentCanvas;
    return {
      isMoveStageKeyHeld,
      stageScale,
      activeTabName,
      baseCanvasImage,
      shouldLockToInitialImage,
      stageDimensions,
      minimumStageScale,
    };
  },
  { memoizeOptions: { resultEqualityCheck: _.isEqual } }
);

const useCanvasWheel = (stageRef: MutableRefObject<Konva.Stage | null>) => {
  const dispatch = useAppDispatch();
  const {
    isMoveStageKeyHeld,
    stageScale,
    activeTabName,
    baseCanvasImage,
    shouldLockToInitialImage,
    stageDimensions,
    minimumStageScale,
  } = useAppSelector(selector);

  return useCallback(
    (e: KonvaEventObject<WheelEvent>) => {
      // stop default scrolling
      if (
        activeTabName !== 'outpainting' ||
        !stageRef.current ||
        isMoveStageKeyHeld ||
        !baseCanvasImage
      )
        return;

      e.evt.preventDefault();

      const cursorPos = stageRef.current.getPointerPosition();

      if (!cursorPos) return;

      const mousePointTo = {
        x: (cursorPos.x - stageRef.current.x()) / stageScale,
        y: (cursorPos.y - stageRef.current.y()) / stageScale,
      };

      let delta = e.evt.deltaY;

      // when we zoom on trackpad, e.evt.ctrlKey is true
      // in that case lets revert direction
      if (e.evt.ctrlKey) {
        delta = -delta;
      }

      const newScale = _.clamp(
        stageScale * CANVAS_SCALE_BY ** delta,
        shouldLockToInitialImage ? minimumStageScale : MIN_CANVAS_SCALE,
        MAX_CANVAS_SCALE
      );

      const newCoordinates = {
        x: cursorPos.x - mousePointTo.x * newScale,
        y: cursorPos.y - mousePointTo.y * newScale,
      };

      if (shouldLockToInitialImage) {
        newCoordinates.x = _.clamp(
          newCoordinates.x,
          stageDimensions.width - Math.floor(baseCanvasImage.width * newScale),
          0
        );
        newCoordinates.y = _.clamp(
          newCoordinates.y,
          stageDimensions.height -
            Math.floor(baseCanvasImage.height * newScale),
          0
        );
      }

      dispatch(setStageScale(newScale));
      dispatch(setStageCoordinates(newCoordinates));
    },
    [
      activeTabName,
      stageRef,
      isMoveStageKeyHeld,
      baseCanvasImage,
      stageScale,
      shouldLockToInitialImage,
      minimumStageScale,
      dispatch,
      stageDimensions.width,
      stageDimensions.height,
    ]
  );
};

export default useCanvasWheel;
