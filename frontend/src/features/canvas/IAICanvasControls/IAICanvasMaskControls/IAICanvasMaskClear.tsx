import { createSelector } from '@reduxjs/toolkit';
import { FaPlus } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import {
  areHotkeysEnabledSelector,
  clearMask,
  currentCanvasSelector,
  GenericCanvasState,
} from 'features/canvas/canvasSlice';

import _ from 'lodash';
import { useHotkeys } from 'react-hotkeys-hook';
import { useToast } from '@chakra-ui/react';

const canvasMaskClearSelector = createSelector(
  [currentCanvasSelector, activeTabNameSelector, areHotkeysEnabledSelector],
  (currentCanvas: GenericCanvasState, activeTabName, areHotkeysEnabled) => {
    const { shouldShowMask, lines } = currentCanvas;

    return {
      shouldShowMask,
      activeTabName,
      isMaskEmpty: lines.length === 0,
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
