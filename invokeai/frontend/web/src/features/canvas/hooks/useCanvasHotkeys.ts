import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';
import {
  clearMask,
  resetCanvasInteractionState,
  setIsMaskEnabled,
  setShouldShowBoundingBox,
  setShouldSnapToGrid,
  setTool,
} from 'features/canvas/store/canvasSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { isEqual } from 'lodash-es';

import { useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
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
      isMaskEnabled,
      shouldSnapToGrid,
    } = canvas;

    return {
      activeTabName,
      isCursorOnCanvas: Boolean(cursorPosition),
      shouldLockBoundingBox,
      shouldShowBoundingBox,
      tool,
      isStaging,
      isMaskEnabled,
      shouldSnapToGrid,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const useInpaintingCanvasHotkeys = () => {
  const dispatch = useAppDispatch();
  const {
    activeTabName,
    shouldShowBoundingBox,
    tool,
    isStaging,
    isMaskEnabled,
    shouldSnapToGrid,
  } = useAppSelector(selector);

  const previousToolRef = useRef<CanvasTool | null>(null);

  const canvasStage = getCanvasStage();

  // Beta Keys
  const handleClearMask = () => dispatch(clearMask());

  useHotkeys(
    ['shift+c'],
    () => {
      handleClearMask();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    []
  );

  const handleToggleEnableMask = () =>
    dispatch(setIsMaskEnabled(!isMaskEnabled));

  useHotkeys(
    ['h'],
    () => {
      handleToggleEnableMask();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [isMaskEnabled]
  );

  useHotkeys(
    ['n'],
    () => {
      dispatch(setShouldSnapToGrid(!shouldSnapToGrid));
    },
    {
      enabled: true,
      preventDefault: true,
    },
    [shouldSnapToGrid]
  );
  //

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
      if (e.repeat) {
        return;
      }

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
