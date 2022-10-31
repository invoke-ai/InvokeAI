import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { useEffect, useRef } from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import { activeTabNameSelector } from '../../../options/optionsSelectors';
import { OptionsState } from '../../../options/optionsSlice';
import {
  InpaintingState,
  setIsDrawing,
  setShouldLockBoundingBox,
  toggleTool,
} from '../inpaintingSlice';

const keyboardEventManagerSelector = createSelector(
  [
    (state: RootState) => state.options,
    (state: RootState) => state.inpainting,
    activeTabNameSelector,
  ],
  (options: OptionsState, inpainting: InpaintingState, activeTabName) => {
    const { shouldShowMask, cursorPosition, shouldLockBoundingBox } =
      inpainting;
    return {
      activeTabName,
      shouldShowMask,
      isCursorOnCanvas: Boolean(cursorPosition),
      shouldLockBoundingBox,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const KeyboardEventManager = () => {
  const dispatch = useAppDispatch();
  const {
    shouldShowMask,
    activeTabName,
    isCursorOnCanvas,
    shouldLockBoundingBox,
  } = useAppSelector(keyboardEventManagerSelector);

  const wasLastEventOverCanvas = useRef<boolean>(false);
  const lastEvent = useRef<KeyboardEvent | null>(null);

  useEffect(() => {
    const listener = (e: KeyboardEvent) => {
      if (
        !['x', ' '].includes(e.key) ||
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
      e.stopPropagation();
      e.preventDefault();
      if (e.repeat) return;
      // cursor is over canvas, we can preventDefault now

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
        case ' ': {
          if (!shouldShowMask) break;

          if (e.type === 'keydown') {
            dispatch(setIsDrawing(false));
          }
          dispatch(setShouldLockBoundingBox(!shouldLockBoundingBox));
          break;
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
  ]);

  return null;
};

export default KeyboardEventManager;
