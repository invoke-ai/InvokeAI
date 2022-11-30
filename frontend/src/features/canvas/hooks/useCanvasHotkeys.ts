import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { useHotkeys } from 'react-hotkeys-hook';
import { activeTabNameSelector } from 'features/options/store/optionsSelectors';
import {
  resetCanvasInteractionState,
  setShouldShowBoundingBox,
  setTool,
} from 'features/canvas/store/canvasSlice';
import { useAppDispatch, useAppSelector } from 'app/store';
import { useRef } from 'react';
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';
import { CanvasTool } from '../store/canvasTypes';
import { getCanvasStage } from '../util/konvaInstanceProvider';

const selector = createSelector(
  [canvasSelector, activeTabNameSelector, isStagingSelector],
  (canvas, activeTabName, isStaging) => {
    const {
      cursorPosition,
      shouldLockBoundingBox,
      shouldShowBoundingBox,
      tool,
    } = canvas;

    return {
      activeTabName,
      isCursorOnCanvas: Boolean(cursorPosition),
      shouldLockBoundingBox,
      shouldShowBoundingBox,
      tool,
      isStaging,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const useInpaintingCanvasHotkeys = () => {
  const dispatch = useAppDispatch();
  const { activeTabName, shouldShowBoundingBox, tool, isStaging } =
    useAppSelector(selector);

  const previousToolRef = useRef<CanvasTool | null>(null);

  const canvasStage = getCanvasStage();

  useHotkeys(
    'esc',
    () => {
      dispatch(resetCanvasInteractionState());
    },
    {
      enabled: () => true,
      preventDefault: true,
    }
  );

  useHotkeys(
    'shift+h',
    () => {
      dispatch(setShouldShowBoundingBox(!shouldShowBoundingBox));
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [activeTabName, shouldShowBoundingBox]
  );

  useHotkeys(
    ['space'],
    (e: KeyboardEvent) => {
      if (e.repeat) return;

      canvasStage?.container().focus();

      if (tool !== 'move') {
        previousToolRef.current = tool;
        dispatch(setTool('move'));
      }

      if (
        tool === 'move' &&
        previousToolRef.current &&
        previousToolRef.current !== 'move'
      ) {
        dispatch(setTool(previousToolRef.current));
        previousToolRef.current = 'move';
      }
    },
    {
      keyup: true,
      keydown: true,
      preventDefault: true,
    },
    [tool, previousToolRef]
  );
};

export default useInpaintingCanvasHotkeys;
