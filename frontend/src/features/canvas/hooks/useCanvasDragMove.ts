import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import { KonvaEventObject } from 'konva/lib/Node';
import _ from 'lodash';
import { useCallback } from 'react';
import {
  currentCanvasSelector,
  GenericCanvasState,
  setStageCoordinates,
} from '../canvasSlice';

const selector = createSelector(
  [currentCanvasSelector, activeTabNameSelector],
  (canvas: GenericCanvasState, activeTabName) => {
    const { isMoveStageKeyHeld } = canvas;
    return {
      isMoveStageKeyHeld,
      activeTabName,
    };
  },
  { memoizeOptions: { resultEqualityCheck: _.isEqual } }
);

const useCanvasDragMove = () => {
  const dispatch = useAppDispatch();
  const { isMoveStageKeyHeld, activeTabName } = useAppSelector(selector);

  return useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!isMoveStageKeyHeld || activeTabName !== 'outpainting') return;
      dispatch(setStageCoordinates(e.target.getPosition()));
    },
    [activeTabName, dispatch, isMoveStageKeyHeld]
  );
};

export default useCanvasDragMove;
