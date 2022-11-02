import { createSelector } from '@reduxjs/toolkit';
import React from 'react';
import { FaPlus } from 'react-icons/fa';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../../app/store';
import IAIIconButton from '../../../../../common/components/IAIIconButton';
import { activeTabNameSelector } from '../../../../options/optionsSelectors';
import { clearMask, InpaintingState } from '../../inpaintingSlice';

import _ from 'lodash';
import { useHotkeys } from 'react-hotkeys-hook';
import { useToast } from '@chakra-ui/react';

const inpaintingMaskClearSelector = createSelector(
  [(state: RootState) => state.inpainting, activeTabNameSelector],
  (inpainting: InpaintingState, activeTabName) => {
    const { shouldShowMask, lines } = inpainting;

    return { shouldShowMask, activeTabName, isMaskEmpty: lines.length === 0 };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function InpaintingMaskClear() {
  const { shouldShowMask, activeTabName, isMaskEmpty } = useAppSelector(
    inpaintingMaskClearSelector
  );

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
      enabled: activeTabName === 'inpainting' && shouldShowMask && !isMaskEmpty,
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
