import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { useEffect, useRef } from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import { OptionsState } from '../../../options/optionsSlice';
import { tabMap } from '../../InvokeTabs';
import {
  InpaintingState,
  setIsMovingBoundingBox,
  toggleIsMovingBoundingBox,
  toggleTool,
} from '../inpaintingSlice';

const keyboardEventManagerSelector = createSelector(
  [(state: RootState) => state.options, (state: RootState) => state.inpainting],
  (options: OptionsState, inpainting: InpaintingState) => {
    const { shouldShowMask, cursorPosition, isMovingBoundingBox } = inpainting;
    return {
      activeTabName: tabMap[options.activeTab],
      shouldShowMask,
      isCursorOnCanvas: Boolean(cursorPosition),
      isMovingBoundingBox,
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
    isMovingBoundingBox,
  } = useAppSelector(keyboardEventManagerSelector);

  const isFirstEvent = useRef<boolean>(true);
  const wasLastEventOverCanvas = useRef<boolean>(false);
  const lastEvent = useRef<KeyboardEvent | null>(null);

  useEffect(() => {
    const listener = (e: KeyboardEvent) => {
      if (
        !['Alt', ' '].includes(e.key) ||
        activeTabName !== 'inpainting' ||
        !shouldShowMask ||
        e.repeat
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

      // cursor is over canvas

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

      e.preventDefault();

      switch (e.key) {
        case 'Alt': {
          dispatch(toggleTool());
          break;
        }
        case ' ': {
          dispatch(setIsMovingBoundingBox(e.type === 'keydown' ? true : false));
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
    isMovingBoundingBox,
  ]);

  return null;
};

export default KeyboardEventManager;
