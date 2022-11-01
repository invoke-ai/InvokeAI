import _ from 'lodash';

const inpaintingCanvasStatusIconsSelector = createSelector(
  (state: RootState) => state.inpainting,
  (inpainting: InpaintingState) => {
    const {
      shouldShowMask,
      shouldInvertMask,
      shouldLockBoundingBox,
      shouldShowBoundingBox,
    } = inpainting;

    return {
      shouldShowMask,
      shouldInvertMask,
      shouldLockBoundingBox,
      shouldShowBoundingBox,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

import { IconButton } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { BiHide, BiShow } from 'react-icons/bi';
import { BsBoundingBox } from 'react-icons/bs';
import { FaLock, FaUnlock } from 'react-icons/fa';
import { MdInvertColors, MdInvertColorsOff } from 'react-icons/md';
import { RootState, useAppSelector } from '../../../app/store';
import { InpaintingState } from './inpaintingSlice';

const InpaintingCanvasStatusIcons = () => {
  const {
    shouldShowMask,
    shouldInvertMask,
    shouldLockBoundingBox,
    shouldShowBoundingBox,
  } = useAppSelector(inpaintingCanvasStatusIconsSelector);

  return (
    <div className="inpainting-alerts">
      <div style={{ pointerEvents: 'none' }}>
        <IconButton
          aria-label="Show/HideMask"
          size="xs"
          variant={'ghost'}
          fontSize={'1rem'}
          data-selected={!shouldShowMask}
          icon={shouldShowMask ? <BiShow /> : <BiHide />}
        />
      </div>
      <div style={{ pointerEvents: 'none' }}>
        <IconButton
          aria-label="Invert Mask"
          variant={'ghost'}
          size="xs"
          fontSize={'1rem'}
          data-selected={shouldInvertMask}
          icon={shouldInvertMask ? <MdInvertColors /> : <MdInvertColorsOff />}
        />
      </div>
      <div style={{ pointerEvents: 'none' }}>
        <IconButton
          aria-label="Bounding Box Lock"
          size="xs"
          variant={'ghost'}
          data-selected={shouldLockBoundingBox}
          icon={shouldLockBoundingBox ? <FaLock /> : <FaUnlock />}
        />
      </div>
      <div style={{ pointerEvents: 'none' }}>
        <IconButton
          aria-label="Bounding Box Lock"
          size="xs"
          variant={'ghost'}
          data-alert={!shouldShowBoundingBox}
          icon={<BsBoundingBox />}
        />
      </div>
    </div>
  );
};

export default InpaintingCanvasStatusIcons;
