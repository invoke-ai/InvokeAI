import {
  Popover,
  PopoverArrow,
  PopoverContent,
  PopoverTrigger,
  Box,
  PopoverHeader,
} from '@chakra-ui/react';
import { ReactNode } from 'react';
import UpscaleOptions from '../options/AdvancedOptions/Upscale/UpscaleOptions';

type UpscalePopoverProps = {
  children: ReactNode;
};

const UpscalePopover = ({ children }: UpscalePopoverProps) => {
  return (
    <Popover trigger={'hover'} closeDelay={300}>
      <PopoverTrigger>
        <Box>{children}</Box>
      </PopoverTrigger>
      <PopoverContent className="popover-content upscale-popover-content">
        <PopoverArrow className="popover-arrow" />
        <PopoverHeader className='popover-header'>Upscale</PopoverHeader>
        <div className="popover-options">
          <UpscaleOptions />
        </div>
      </PopoverContent>
    </Popover>
  );
};

export default UpscalePopover;
