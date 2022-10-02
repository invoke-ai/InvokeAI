import {
  Popover,
  PopoverArrow,
  PopoverContent,
  PopoverTrigger,
  Box,
  PopoverHeader,
} from '@chakra-ui/react';
import { ReactNode } from 'react';
import FaceRestoreOptions from '../options/AdvancedOptions/FaceRestore/FaceRestoreOptions';

type FaceRestorePopoverProps = {
  children: ReactNode;
};

const FaceRestorePopover = ({ children }: FaceRestorePopoverProps) => {
  return (
    <Popover trigger={'hover'} closeDelay={300}>
      <PopoverTrigger>
        <Box>{children}</Box>
      </PopoverTrigger>
      <PopoverContent className="popover-content face-restore-popover-content">
        <PopoverArrow className="popover-arrow" />
        <PopoverHeader className="popover-header">Face Restore</PopoverHeader>
        <div className="popover-options">
          <FaceRestoreOptions />
        </div>
      </PopoverContent>
    </Popover>
  );
};

export default FaceRestorePopover;
