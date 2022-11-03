import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { useEffect, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import { activeTabNameSelector } from '../../options/optionsSelectors';
import { OptionsState } from '../../options/optionsSlice';
import {
  InpaintingState,
  setIsMoveBoundingBoxKeyHeld,
  setIsMoveStageKeyHeld,
  setShouldLockBoundingBox,
  toggleShouldLockBoundingBox,
  toggleTool,
} from './inpaintingSlice';

const canvasHotkeysSelector = createSelector(
  [
    (state: RootState) => state.options,
    (state: RootState) => state.inpainting,
    activeTabNameSelector,
  ],
  (options: OptionsState, inpainting: InpaintingState, activeTabName) => {
    const {
      shouldShowMask,
      cursorPosition,
      shouldLockBoundingBox,
      shouldShowBoundingBox,
    } = inpainting;
    return {
      activeTabName,
      shouldShowMask,
      isCursorOnCanvas: Boolean(cursorPosition),
      shouldLockBoundingBox,
      shouldShowBoundingBox,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const useCanvasHotkeys = () => {
  const dispatch = useAppDispatch();
  const {
    shouldShowMask,
    activeTabName,
    isCursorOnCanvas,
    shouldLockBoundingBox,
    shouldShowBoundingBox,
  } = useAppSelector(canvasHotkeysSelector);

  const wasLastEventOverCanvas = useRef<boolean>(false);
  const lastEvent = useRef<KeyboardEvent | null>(null);

  //  Toggle lock bounding box
  useHotkeys(
    'shift+q',
    (e: KeyboardEvent) => {
      e.preventDefault();
      dispatch(toggleShouldLockBoundingBox());
    },
    {
      enabled: activeTabName === 'inpainting' && shouldShowMask,
    },
    [activeTabName, shouldShowMask]
  );

  // Manages hold-style keyboard shortcuts
  useEffect(() => {
    const listener = (e: KeyboardEvent) => {
      if (
        !['x', 'q', 'w'].includes(e.key) ||
        activeTabName !== 'inpainting' ||
        !shouldShowMask
      ) {
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
        case 'q': {
          if (!shouldShowMask || !shouldShowBoundingBox) break;
          dispatch(setIsMoveBoundingBoxKeyHeld(e.type === 'keydown'));
          dispatch(setShouldLockBoundingBox(e.type !== 'keydown'));
          break;
        }
        case 'w': {
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
  ]);
};

export default useCanvasHotkeys;
