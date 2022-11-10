import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import Konva from 'konva';
import { KonvaEventObject } from 'konva/lib/Node';
import _ from 'lodash';
import { MutableRefObject, useCallback } from 'react';
import {
  currentCanvasSelector,
  GenericCanvasState,
  setStageCoordinates,
  setStageScale,
} from '../canvasSlice';
import {
  CANVAS_SCALE_BY,
  MAX_CANVAS_SCALE,
  MIN_CANVAS_SCALE,
} from '../util/constants';

const selector = createSelector(
  [activeTabNameSelector, currentCanvasSelector],
  (activeTabName, canvas: GenericCanvasState) => {
    const { isMoveStageKeyHeld, stageScale } = canvas;
    return {
      isMoveStageKeyHeld,
      stageScale,
      activeTabName,
    };
  },
  { memoizeOptions: { resultEqualityCheck: _.isEqual } }
);

const useCanvasWheel = (stageRef: MutableRefObject<Konva.Stage | null>) => {
  const dispatch = useAppDispatch();
  const { isMoveStageKeyHeld, stageScale, activeTabName } =
    useAppSelector(selector);

  return useCallback(
    (e: KonvaEventObject<WheelEvent>) => {
      // stop default scrolling
      if (activeTabName !== 'outpainting') return;

      e.evt.preventDefault();

      // const oldScale = stageRef.current.scaleX();
      if (!stageRef.current || isMoveStageKeyHeld) return;

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

      const newPos = {
        x: cursorPos.x - mousePointTo.x * newScale,
        y: cursorPos.y - mousePointTo.y * newScale,
      };

      dispatch(setStageScale(newScale));
      dispatch(setStageCoordinates(newPos));
    },
    [activeTabName, dispatch, isMoveStageKeyHeld, stageRef, stageScale]
  );
};

export default useCanvasWheel;
