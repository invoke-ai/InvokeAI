import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { useEffect, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import { OptionsState } from 'features/options/optionsSlice';
import {
  areHotkeysEnabledSelector,
  setIsMoveBoundingBoxKeyHeld,
  setIsMoveStageKeyHeld,
  setShouldLockBoundingBox,
  toggleShouldLockBoundingBox,
  toggleTool,
} from 'features/canvas/canvasSlice';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import { currentCanvasSelector, GenericCanvasState } from '../canvasSlice';

const inpaintingCanvasHotkeysSelector = createSelector(
  [
    (state: RootState) => state.options,
    currentCanvasSelector,
    activeTabNameSelector,
    areHotkeysEnabledSelector,
  ],
  (
    options: OptionsState,
    currentCanvas: GenericCanvasState,
    activeTabName,
    areHotkeysEnabled
  ) => {
    const {
      shouldShowMask,
      cursorPosition,
      shouldLockBoundingBox,
      shouldShowBoundingBox,
    } = currentCanvas;
    return {
      activeTabName,
      shouldShowMask,
      isCursorOnCanvas: Boolean(cursorPosition),
      shouldLockBoundingBox,
      shouldShowBoundingBox,
      areHotkeysEnabled,
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
  const {
    shouldShowMask,
    activeTabName,
    isCursorOnCanvas,
    shouldLockBoundingBox,
    shouldShowBoundingBox,
    areHotkeysEnabled,
  } = useAppSelector(inpaintingCanvasHotkeysSelector);

  const wasLastEventOverCanvas = useRef<boolean>(false);
  const lastEvent = useRef<KeyboardEvent | null>(null);

  //  Toggle lock bounding box
  useHotkeys(
    'shift+w',
    (e: KeyboardEvent) => {
      e.preventDefault();
      dispatch(toggleShouldLockBoundingBox());
    },
    {
      enabled: areHotkeysEnabled,
    },
    [activeTabName, shouldShowMask]
  );

  // Manages hold-style keyboard shortcuts
  useEffect(() => {
    const listener = (e: KeyboardEvent) => {
      if (!['x', 'w', 'q'].includes(e.key) || !areHotkeysEnabled) {
        return;
      }

      // cursor is NOT over canvas
      if (!isCursorOnCanvas) {
        if (!lastEvent.current) {
          lastEvent.current = e;
        }

        wasLastEventOverCanvas.current = false;
        return;
      }

      // cursor is over canvas, we can preventDefault now
      e.stopPropagation();
      e.preventDefault();
      if (e.repeat) return;

      // if this is the first event
      if (!lastEvent.current) {
        wasLastEventOverCanvas.current = true;
        lastEvent.current = e;
      }

      if (!wasLastEventOverCanvas.current && e.type === 'keyup') {
        wasLastEventOverCanvas.current = true;
        lastEvent.current = e;
        return;
      }

      switch (e.key) {
        case 'x': {
          dispatch(toggleTool());
          break;
        }
        case 'w': {
          if (!shouldShowMask || !shouldShowBoundingBox) break;
          dispatch(setIsMoveBoundingBoxKeyHeld(e.type === 'keydown'));
          dispatch(setShouldLockBoundingBox(e.type !== 'keydown'));
          break;
        }
        case 'q': {
          if (!shouldShowMask) break;
          dispatch(setIsMoveStageKeyHeld(e.type === 'keydown'));
        }
      }

      lastEvent.current = e;
      wasLastEventOverCanvas.current = true;
    };

    document.addEventListener('keydown', listener);
    document.addEventListener('keyup', listener);

    return () => {
      document.removeEventListener('keydown', listener);
      document.removeEventListener('keyup', listener);
    };
  }, [
    dispatch,
    activeTabName,
    shouldShowMask,
    isCursorOnCanvas,
    shouldLockBoundingBox,
    shouldShowBoundingBox,
    areHotkeysEnabled,
  ]);
};

export default useInpaintingCanvasHotkeys;
