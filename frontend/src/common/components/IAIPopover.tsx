import {
  Popover,
  PopoverArrow,
  PopoverContent,
  PopoverTrigger,
  BoxProps,
} from '@chakra-ui/react';
import { PopoverProps } from '@chakra-ui/react';
import { ReactNode } from 'react';

type IAIPopoverProps = PopoverProps & {
  triggerComponent: ReactNode;
  triggerContainerProps?: BoxProps;
  children: ReactNode;
  styleClass?: string;
  hasArrow?: boolean;
};

const IAIPopover = (props: IAIPopoverProps) => {
  const {
    triggerComponent,
    children,
    styleClass,
    hasArrow = true,
    ...rest
  } = props;

  return (
    <Popover {...rest}>
      <PopoverTrigger>{triggerComponent}</PopoverTrigger>
      <PopoverContent className={`invokeai__popover-content ${styleClass}`}>
        {hasArrow && <PopoverArrow className="invokeai__popover-arrow" />}
        {children}
      </PopoverContent>
    </Popover>
  );
};

export default IAIPopover;
