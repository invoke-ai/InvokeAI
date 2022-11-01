import React from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { BiHide, BiShow } from 'react-icons/bi';
import { createSelector } from 'reselect';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../../app/store';
import IAIIconButton from '../../../../../common/components/IAIIconButton';
import { activeTabNameSelector } from '../../../../options/optionsSelectors';
import { InpaintingState, setShouldShowMask } from '../../inpaintingSlice';

import _ from 'lodash';

const inpaintingMaskVisibilitySelector = createSelector(
  [(state: RootState) => state.inpainting, activeTabNameSelector],
  (inpainting: InpaintingState, activeTabName) => {
    const { shouldShowMask } = inpainting;

    return { shouldShowMask, activeTabName };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function InpaintingMaskVisibilityControl() {
  const dispatch = useAppDispatch();

  const { shouldShowMask, activeTabName } = useAppSelector(
    inpaintingMaskVisibilitySelector
  );

  const handleToggleShouldShowMask = () =>
    dispatch(setShouldShowMask(!shouldShowMask));
  // Hotkeys
  // Show/hide mask
  useHotkeys(
    'h',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleToggleShouldShowMask();
    },
    {
      enabled: activeTabName === 'inpainting',
    },
    [activeTabName, shouldShowMask]
  );
  return (
    <IAIIconButton
      aria-label="Hide/Show Mask (H)"
      tooltip="Hide/Show Mask (H)"
      data-selected={!shouldShowMask}
      icon={shouldShowMask ? <BiShow size={22} /> : <BiHide size={22} />}
      onClick={handleToggleShouldShowMask}
    />
  );
}
