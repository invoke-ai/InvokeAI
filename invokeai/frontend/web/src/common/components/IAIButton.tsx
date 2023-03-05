import {
  Button,
  ButtonProps,
  forwardRef,
  Tooltip,
  TooltipProps,
} from '@chakra-ui/react';
import { ReactNode } from 'react';

export interface IAIButtonProps extends ButtonProps {
  tooltip?: string;
  tooltipProps?: Omit<TooltipProps, 'children'>;
  isChecked?: boolean;
  children: ReactNode;
}

const IAIButton = forwardRef((props: IAIButtonProps, forwardedRef) => {
  const { children, tooltip = '', tooltipProps, isChecked, ...rest } = props;
  return (
    <Tooltip label={tooltip} {...tooltipProps}>
      <Button ref={forwardedRef} aria-checked={isChecked} {...rest}>
        {children}
      </Button>
    </Tooltip>
  );
});

export default IAIButton;
