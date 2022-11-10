import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { useHotkeys } from 'react-hotkeys-hook';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import { OptionsState } from 'features/options/optionsSlice';
import {
  CanvasTool,
  setShouldShowBoundingBox,
  setTool,
  toggleShouldLockBoundingBox,
} from 'features/canvas/canvasSlice';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import { currentCanvasSelector, GenericCanvasState } from '../canvasSlice';
import { useRef } from 'react';

const inpaintingCanvasHotkeysSelector = createSelector(
  [
    (state: RootState) => state.options,
    currentCanvasSelector,
    activeTabNameSelector,
  ],
  (options: OptionsState, currentCanvas: GenericCanvasState, activeTabName) => {
    const {
      isMaskEnabled,
      cursorPosition,
      shouldLockBoundingBox,
      shouldShowBoundingBox,
      tool,
    } = currentCanvas;

    return {
      activeTabName,
      isMaskEnabled,
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
  const { isMaskEnabled, activeTabName, shouldShowBoundingBox, tool } =
    useAppSelector(inpaintingCanvasHotkeysSelector);

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
