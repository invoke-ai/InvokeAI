import {
  Box,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { Feature, useFeatureHelpInfo } from 'app/features';
import { useAppSelector } from 'app/store/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';
import { SystemState } from 'features/system/store/systemSlice';
import { memo, ReactElement } from 'react';

type GuideProps = {
  children: ReactElement;
  feature: Feature;
};

const guidePopoverSelector = createSelector(
  systemSelector,
  (system: SystemState) => system.shouldDisplayGuides
);

const GuidePopover = ({ children, feature }: GuideProps) => {
  const shouldDisplayGuides = useAppSelector(guidePopoverSelector);
  const { text } = useFeatureHelpInfo(feature);

  if (!shouldDisplayGuides) return null;

  return (
    <Popover trigger="hover" isLazy>
      <PopoverTrigger>
        <Box>{children}</Box>
      </PopoverTrigger>
      <PopoverContent
        maxWidth="400px"
        onClick={(e) => e.preventDefault()}
        cursor="initial"
      >
        <PopoverArrow />
        <PopoverBody>{text}</PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

export default memo(GuidePopover);
