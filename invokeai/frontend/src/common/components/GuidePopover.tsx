import {
  Popover,
  PopoverArrow,
  PopoverContent,
  PopoverTrigger,
  Box,
} from '@chakra-ui/react';
import { SystemState } from 'features/system/store/systemSlice';
import { useAppSelector } from 'app/storeHooks';
import { createSelector } from '@reduxjs/toolkit';
import { ReactElement } from 'react';
import { Feature, useFeatureHelpInfo } from 'app/features';
import { systemSelector } from 'features/system/store/systemSelectors';

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
    <Popover trigger={'hover'}>
      <PopoverTrigger>
        <Box>{children}</Box>
      </PopoverTrigger>
      <PopoverContent
        className={`guide-popover-content`}
        maxWidth="400px"
        onClick={(e) => e.preventDefault()}
        cursor={'initial'}
      >
        <PopoverArrow className="guide-popover-arrow" />
        <div className="guide-popover-guide-content">{text}</div>
      </PopoverContent>
    </Popover>
  );
};

export default GuidePopover;
