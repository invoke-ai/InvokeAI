import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { useEffect, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import { OptionsState } from 'features/options/optionsSlice';
import {
  areHotkeysEnabledSelector,
  CanvasTool,
  setIsMoveBoundingBoxKeyHeld,
  setIsMoveStageKeyHeld,
  setShouldLockBoundingBox,
  setShouldShowBoundingBox,
  setTool,
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
      tool,
    } = currentCanvas;
    return {
      activeTabName,
      shouldShowMask,
      isCursorOnCanvas: Boolean(cursorPosition),
      shouldLockBoundingBox,
      shouldShowBoundingBox,
      areHotkeysEnabled,
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
  const {
    shouldShowMask,
    activeTabName,
    isCursorOnCanvas,
    shouldLockBoundingBox,
    shouldShowBoundingBox,
    areHotkeysEnabled,
    tool,
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

  useHotkeys(
    'shift+h',
    (e: KeyboardEvent) => {
      e.preventDefault();
      dispatch(setShouldShowBoundingBox(!shouldShowBoundingBox));
    },
    {
      enabled: areHotkeysEnabled,
    },
    [activeTabName, shouldShowBoundingBox]
  );

  useEffect(() => {
    let previousTool: CanvasTool = tool;

    const keyupListener = (e: KeyboardEvent) => {
      if (e.key !== ' ' || !isCursorOnCanvas) return;
      e.stopPropagation();
      e.preventDefault();

      dispatch(setTool(previousTool));
      window.removeEventListener('keyup', keyupListener);
      window.addEventListener('keydown', keydownListener);
    };

    const keydownListener = (e: KeyboardEvent) => {
      if (e.key !== ' ' || !areHotkeysEnabled || !isCursorOnCanvas) return;
      e.stopPropagation();
      e.preventDefault();
      previousTool = tool;

      if (tool !== 'move') {
        dispatch(setTool('move'));
        window.addEventListener('keyup', keyupListener);
        window.removeEventListener('keydown', keydownListener);
      }
    };

    document.addEventListener('keydown', keydownListener);

    return () => {
      document.removeEventListener('keydown', keydownListener);
      document.removeEventListener('keyup', keyupListener);
    };
  }, [areHotkeysEnabled, dispatch, isCursorOnCanvas, tool]);

  // // Manages hold-style keyboard shortcuts
  // useEffect(() => {
  //   const listener = (e: KeyboardEvent) => {
  //     if (!['x', 'w', 'q'].includes(e.key) || !areHotkeysEnabled) {
  //       return;
  //     }

  //     // cursor is NOT over canvas
  //     if (!isCursorOnCanvas) {
  //       if (!lastEvent.current) {
  //         lastEvent.current = e;
  //       }

  //       wasLastEventOverCanvas.current = false;
  //       return;
  //     }

  //     // cursor is over canvas, we can preventDefault now
  //     e.stopPropagation();
  //     e.preventDefault();
  //     if (e.repeat) return;

  //     // if this is the first event
  //     if (!lastEvent.current) {
  //       wasLastEventOverCanvas.current = true;
  //       lastEvent.current = e;
  //     }

  //     if (!wasLastEventOverCanvas.current && e.type === 'keyup') {
  //       wasLastEventOverCanvas.current = true;
  //       lastEvent.current = e;
  //       return;
  //     }

  //     switch (e.key) {
  //       case 'x': {
  //         dispatch(toggleTool());
  //         break;
  //       }
  //       case 'w': {
  //         if (!shouldShowMask || !shouldShowBoundingBox) break;
  //         dispatch(setIsMoveBoundingBoxKeyHeld(e.type === 'keydown'));
  //         dispatch(setShouldLockBoundingBox(e.type !== 'keydown'));
  //         break;
  //       }
  //       case 'q': {
  //         if (!shouldShowMask || activeTabName === 'inpainting') break;
  //         dispatch(setIsMoveStageKeyHeld(e.type === 'keydown'));
  //       }
  //     }

  //     lastEvent.current = e;
  //     wasLastEventOverCanvas.current = true;
  //   };

  //   document.addEventListener('keydown', listener);
  //   document.addEventListener('keyup', listener);

  //   return () => {
  //     document.removeEventListener('keydown', listener);
  //     document.removeEventListener('keyup', listener);
  //   };
  // }, [
  //   dispatch,
  //   activeTabName,
  //   shouldShowMask,
  //   isCursorOnCanvas,
  //   shouldLockBoundingBox,
  //   shouldShowBoundingBox,
  //   areHotkeysEnabled,
  // ]);
};

export default useInpaintingCanvasHotkeys;
