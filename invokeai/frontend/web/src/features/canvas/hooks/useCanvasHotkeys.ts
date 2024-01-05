import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  resetCanvasInteractionState,
  resetToolInteractionState,
} from 'features/canvas/store/canvasNanostore';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import {
  clearMask,
  setIsMaskEnabled,
  setShouldShowBoundingBox,
  setShouldSnapToGrid,
  setTool,
} from 'features/canvas/store/canvasSlice';
import type { CanvasTool } from 'features/canvas/store/canvasTypes';
import { getCanvasStage } from 'features/canvas/util/konvaInstanceProvider';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useCallback, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';

const useInpaintingCanvasHotkeys = () => {
  const dispatch = useAppDispatch();
  const activeTabName = useAppSelector(activeTabNameSelector);
  const shouldShowBoundingBox = useAppSelector(
    (s) => s.canvas.shouldShowBoundingBox
  );
  const tool = useAppSelector((s) => s.canvas.tool);
  const isStaging = useAppSelector(isStagingSelector);
  const isMaskEnabled = useAppSelector((s) => s.canvas.isMaskEnabled);
  const shouldSnapToGrid = useAppSelector((s) => s.canvas.shouldSnapToGrid);
  const previousToolRef = useRef<CanvasTool | null>(null);
  const canvasStage = getCanvasStage();

  // Beta Keys
  const handleClearMask = useCallback(() => dispatch(clearMask()), [dispatch]);

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
      resetCanvasInteractionState();
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
        resetToolInteractionState();
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
