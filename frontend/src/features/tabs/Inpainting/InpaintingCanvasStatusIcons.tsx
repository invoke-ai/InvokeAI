import _ from 'lodash';

const inpaintingCanvasStatusIconsSelector = createSelector(
  (state: RootState) => state.inpainting,
  (inpainting: InpaintingState) => {
    const {
      shouldShowMask,
      shouldInvertMask,
      shouldLockBoundingBox,
      shouldShowBoundingBox,
      boundingBoxDimensions,
    } = inpainting;

    return {
      shouldShowMask,
      shouldInvertMask,
      shouldLockBoundingBox,
      shouldShowBoundingBox,
      isBoundingBoxTooSmall:
        boundingBoxDimensions.width < 512 || boundingBoxDimensions.height < 512,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

import { ButtonGroup, IconButton } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { BiHide, BiShow } from 'react-icons/bi';
import { GiResize } from 'react-icons/gi';
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
    isBoundingBoxTooSmall,
  } = useAppSelector(inpaintingCanvasStatusIconsSelector);

  return (
    <div className="inpainting-alerts">
      <ButtonGroup isAttached>
        <IconButton
          aria-label="Show/HideMask"
          size="xs"
          variant={'ghost'}
          fontSize={'1rem'}
          data-selected={!shouldShowMask}
          icon={shouldShowMask ? <BiShow /> : <BiHide />}
        />
        <IconButton
          aria-label="Invert Mask"
          variant={'ghost'}
          size="xs"
          fontSize={'1rem'}
          data-selected={shouldInvertMask}
          icon={shouldInvertMask ? <MdInvertColors /> : <MdInvertColorsOff />}
        />
        <IconButton
          aria-label="Bounding Box Lock"
          size="xs"
          variant={'ghost'}
          data-selected={shouldLockBoundingBox}
          icon={shouldLockBoundingBox ? <FaLock /> : <FaUnlock />}
        />
        <IconButton
          aria-label="Bounding Box Lock"
          size="xs"
          variant={'ghost'}
          data-alert={!shouldShowBoundingBox}
          icon={<BsBoundingBox />}
        />
        <IconButton
          aria-label="Under 512x512"
          size="xs"
          variant={'ghost'}
          data-alert={isBoundingBoxTooSmall}
          icon={<GiResize />}
        />
      </ButtonGroup>
    </div>
  );
};

export default InpaintingCanvasStatusIcons;
