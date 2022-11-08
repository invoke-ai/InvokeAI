import { createSelector } from '@reduxjs/toolkit';
import { FaPlus } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import {
  areHotkeysEnabledSelector,
  clearMask,
  currentCanvasSelector,
  InpaintingCanvasState,
  isCanvasMaskLine,
  OutpaintingCanvasState,
} from 'features/canvas/canvasSlice';

import _ from 'lodash';
import { useHotkeys } from 'react-hotkeys-hook';
import { useToast } from '@chakra-ui/react';

const canvasMaskClearSelector = createSelector(
  [currentCanvasSelector, activeTabNameSelector, areHotkeysEnabledSelector],
  (currentCanvas, activeTabName, areHotkeysEnabled) => {
    const { shouldShowMask, objects } = currentCanvas as
      | InpaintingCanvasState
      | OutpaintingCanvasState;

    return {
      shouldShowMask,
      activeTabName,
      isMaskEmpty: objects.filter(isCanvasMaskLine).length === 0,
      areHotkeysEnabled,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function IAICanvasMaskClear() {
  const { shouldShowMask, activeTabName, isMaskEmpty, areHotkeysEnabled } =
    useAppSelector(canvasMaskClearSelector);

  const dispatch = useAppDispatch();
  const toast = useToast();

  const handleClearMask = () => {
    dispatch(clearMask());
  };

  // Clear mask
  useHotkeys(
    'shift+c',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleClearMask();
      toast({
        title: 'Mask Cleared',
        status: 'success',
        duration: 2500,
        isClosable: true,
      });
    },
    {
      enabled: areHotkeysEnabled && !isMaskEmpty,
    },
    [activeTabName, isMaskEmpty, shouldShowMask]
  );
  return (
    <IAIIconButton
      aria-label="Clear Mask (Shift+C)"
      tooltip="Clear Mask (Shift+C)"
      icon={<FaPlus size={20} style={{ transform: 'rotate(45deg)' }} />}
      onClick={handleClearMask}
      isDisabled={isMaskEmpty || !shouldShowMask}
    />
  );
}
