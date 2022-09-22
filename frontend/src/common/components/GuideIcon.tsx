import { Box, forwardRef, Icon } from '@chakra-ui/react';
import { IconType } from 'react-icons';
import { MdHelp } from 'react-icons/md';
import { Guides } from '../../app/guides';
import GuidePopover from './GuidePopover';

type GuideIconProps = {
  feature: keyof typeof Guides;
  icon?: IconType;
};

const GuideIcon = forwardRef(
  ({ feature, icon = MdHelp }: GuideIconProps, ref) => (
    <GuidePopover feature={feature}>
      <Box ref={ref}>
        <Icon as={icon} />
      </Box>
    </GuidePopover>
  )
);

export default GuideIcon;
