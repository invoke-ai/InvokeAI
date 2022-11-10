import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import { KonvaEventObject } from 'konva/lib/Node';
import _ from 'lodash';
import { useCallback } from 'react';
import {
  currentCanvasSelector,
  setIsMovingStage,
  setStageCoordinates,
} from '../canvasSlice';

const selector = createSelector(
  [currentCanvasSelector, activeTabNameSelector],
  (canvas, activeTabName) => {
    const { tool } = canvas;
    return {
      tool,

      activeTabName,
    };
  },
  { memoizeOptions: { resultEqualityCheck: _.isEqual } }
);

const useCanvasDrag = () => {
  const dispatch = useAppDispatch();
  const { tool, activeTabName } = useAppSelector(selector);

  return {
    handleDragStart: useCallback(() => {
      if (tool !== 'move' || activeTabName !== 'outpainting') return;
      dispatch(setIsMovingStage(true));
    }, [activeTabName, dispatch, tool]),
    handleDragMove: useCallback(
      (e: KonvaEventObject<MouseEvent>) => {
        if (tool !== 'move' || activeTabName !== 'outpainting') return;
        dispatch(setStageCoordinates(e.target.getPosition()));
      },
      [activeTabName, dispatch, tool]
    ),
    handleDragEnd: useCallback(() => {
      if (tool !== 'move' || activeTabName !== 'outpainting') return;
      dispatch(setIsMovingStage(false));
    }, [activeTabName, dispatch, tool]),
  };
};

export default useCanvasDrag;
