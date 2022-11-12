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
import { stageRef } from '../IAICanvas';

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
      scopes: ['inpainting', 'outpainting'],
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
      scopes: ['inpainting', 'outpainting'],
    },
    [activeTabName, shouldShowBoundingBox]
  );

  useHotkeys(
    ['space'],
    (e: KeyboardEvent) => {
      if (e.repeat) return;

      stageRef.current?.container().focus();

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
      scopes: ['inpainting', 'outpainting'],
    },
    [tool, previousToolRef]
  );
};

export default useInpaintingCanvasHotkeys;
