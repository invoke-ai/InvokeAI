import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import Konva from 'konva';
import { KonvaEventObject } from 'konva/lib/Node';
import _ from 'lodash';
import { MutableRefObject, useCallback } from 'react';
import {
  canvasSelector,
  initialCanvasImageSelector,
} from 'features/canvas/store/canvasSelectors';
import {
  setStageCoordinates,
  setStageScale,
} from 'features/canvas/store/canvasSlice';
import {
  CANVAS_SCALE_BY,
  MAX_CANVAS_SCALE,
  MIN_CANVAS_SCALE,
} from '../util/constants';

const selector = createSelector(
  [activeTabNameSelector, canvasSelector, initialCanvasImageSelector],
  (activeTabName, canvas, initialCanvasImage) => {
    const {
      isMoveStageKeyHeld,
      stageScale,
      stageDimensions,
      minimumStageScale,
    } = canvas;
    return {
      isMoveStageKeyHeld,
      stageScale,
      activeTabName,
      initialCanvasImage,
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
    initialCanvasImage,
    stageDimensions,
    minimumStageScale,
  } = useAppSelector(selector);

  return useCallback(
    (e: KonvaEventObject<WheelEvent>) => {
      // stop default scrolling
      if (!stageRef.current || isMoveStageKeyHeld || !initialCanvasImage)
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
        MIN_CANVAS_SCALE,
        MAX_CANVAS_SCALE
      );

      const newCoordinates = {
        x: cursorPos.x - mousePointTo.x * newScale,
        y: cursorPos.y - mousePointTo.y * newScale,
      };

      dispatch(setStageScale(newScale));
      dispatch(setStageCoordinates(newCoordinates));
    },
    [stageRef, isMoveStageKeyHeld, initialCanvasImage, stageScale, dispatch]
  );
};

export default useCanvasWheel;
