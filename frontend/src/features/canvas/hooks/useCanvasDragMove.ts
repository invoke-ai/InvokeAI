import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import { KonvaEventObject } from 'konva/lib/Node';
import _ from 'lodash';
import { useCallback } from 'react';
import {
  currentCanvasSelector,
  isStagingSelector,
  setIsMovingStage,
  setStageCoordinates,
} from '../canvasSlice';

const selector = createSelector(
  [currentCanvasSelector, isStagingSelector, activeTabNameSelector],
  (canvas, isStaging, activeTabName) => {
    const { tool } = canvas;
    return {
      tool,
      isStaging,
      activeTabName,
    };
  },
  { memoizeOptions: { resultEqualityCheck: _.isEqual } }
);

const useCanvasDrag = () => {
  const dispatch = useAppDispatch();
  const { tool, activeTabName, isStaging } = useAppSelector(selector);

  return {
    handleDragStart: useCallback(() => {
      if (!(tool === 'move' || isStaging)) return;
      dispatch(setIsMovingStage(true));
    }, [dispatch, isStaging, tool]),

    handleDragMove: useCallback(
      (e: KonvaEventObject<MouseEvent>) => {
        if (!(tool === 'move' || isStaging)) return;
        dispatch(setStageCoordinates(e.target.getPosition()));
      },
      [dispatch, isStaging, tool]
    ),

    handleDragEnd: useCallback(() => {
      if (!(tool === 'move' || isStaging)) return;
      dispatch(setIsMovingStage(false));
    }, [dispatch, isStaging, tool]),
  };
};

export default useCanvasDrag;
