import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { useHotkeys } from 'react-hotkeys-hook';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import {
  CanvasTool,
  setShouldShowBoundingBox,
  setTool,
  toggleShouldLockBoundingBox,
} from 'features/canvas/canvasSlice';
import { useAppDispatch, useAppSelector } from 'app/store';
import { currentCanvasSelector } from '../canvasSlice';
import { useRef } from 'react';

const inpaintingCanvasHotkeysSelector = createSelector(
  [currentCanvasSelector, activeTabNameSelector],
  (currentCanvas, activeTabName) => {
    const {
      cursorPosition,
      shouldLockBoundingBox,
      shouldShowBoundingBox,
      tool,
    } = currentCanvas;

    return {
      activeTabName,
      isCursorOnCanvas: Boolean(cursorPosition),
      shouldLockBoundingBox,
      shouldShowBoundingBox,
      tool,
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
  const { activeTabName, shouldShowBoundingBox, tool } = useAppSelector(
    inpaintingCanvasHotkeysSelector
  );

  const previousToolRef = useRef<CanvasTool | null>(null);
  //  Toggle lock bounding box
  useHotkeys(
    'shift+w',
    (e: KeyboardEvent) => {
      e.preventDefault();
      dispatch(toggleShouldLockBoundingBox());
    },
    {
      enabled: true,
    },
    [activeTabName]
  );

  useHotkeys(
    'shift+h',
    (e: KeyboardEvent) => {
      e.preventDefault();
      dispatch(setShouldShowBoundingBox(!shouldShowBoundingBox));
    },
    {
      enabled: true,
    },
    [activeTabName, shouldShowBoundingBox]
  );

  useHotkeys(
    ['space'],
    (e: KeyboardEvent) => {
      if (e.repeat) return;

      if (tool !== 'move') {
        previousToolRef.current = tool;
        dispatch(setTool('move'));
      }
    },
    { keyup: false, keydown: true },
    [tool, previousToolRef]
  );

  useHotkeys(
    ['space'],
    (e: KeyboardEvent) => {
      if (e.repeat) return;

      if (
        tool === 'move' &&
        previousToolRef.current &&
        previousToolRef.current !== 'move'
      ) {
        dispatch(setTool(previousToolRef.current));
        previousToolRef.current = 'move';
      }
    },
    { keyup: true, keydown: false },
    [tool, previousToolRef]
  );
};

export default useInpaintingCanvasHotkeys;
