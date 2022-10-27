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
  toggleIsMovingBoundingBox,
  toggleTool,
} from '../inpaintingSlice';

const keyboardEventManagerSelector = createSelector(
  [(state: RootState) => state.options, (state: RootState) => state.inpainting],
  (options: OptionsState, inpainting: InpaintingState) => {
    const { shouldShowMask, cursorPosition } = inpainting;
    return {
      activeTabName: tabMap[options.activeTab],
      shouldShowMask,
      isCursorOnCanvas: Boolean(cursorPosition),
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
  const { shouldShowMask, activeTabName, isCursorOnCanvas } = useAppSelector(
    keyboardEventManagerSelector
  );

  const isFirstEvent = useRef<boolean>(true);
  const wasLastEventOverCanvas = useRef<boolean>(false);

  useEffect(() => {
    const listener = (e: KeyboardEvent) => {
      if (!isCursorOnCanvas) {
        wasLastEventOverCanvas.current = false;

        if (isFirstEvent.current) {
          isFirstEvent.current = false;
        }

        return;
      }

      if (isFirstEvent.current) {
        wasLastEventOverCanvas.current = true;
        isFirstEvent.current = false;
      }

      if (
        !['Alt', ' '].includes(e.key) ||
        activeTabName !== 'inpainting' ||
        !shouldShowMask ||
        e.repeat
      ) {
        return;
      }

      if (!wasLastEventOverCanvas.current) {
        wasLastEventOverCanvas.current = true;
        return;
      }

      e.preventDefault();

      switch (e.key) {
        case 'Alt': {
          dispatch(toggleTool());
          break;
        }
        case ' ': {
          dispatch(toggleIsMovingBoundingBox());
          break;
        }
      }
    };

    console.log('adding listeners');
    document.addEventListener('keydown', listener);
    document.addEventListener('keyup', listener);

    return () => {
      document.removeEventListener('keydown', listener);
      document.removeEventListener('keyup', listener);
    };
  }, [dispatch, activeTabName, shouldShowMask, isCursorOnCanvas]);

  return null;
};

export default KeyboardEventManager;
