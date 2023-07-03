import {
  Button,
  ButtonProps,
  forwardRef,
  Tooltip,
  TooltipProps,
} from '@chakra-ui/react';
import { memo, ReactNode } from 'react';

export interface IAIButtonProps extends ButtonProps {
  tooltip?: string;
  tooltipProps?: Omit<TooltipProps, 'children'>;
  isChecked?: boolean;
  children: ReactNode;
}

const IAIButton = forwardRef((props: IAIButtonProps, forwardedRef) => {
  const {
    children,
    tooltip = '',
    tooltipProps: { placement = 'top', hasArrow = true, ...tooltipProps } = {},
    isChecked,
    ...rest
  } = props;
  return (
    <Tooltip
      label={tooltip}
      placement={placement}
      hasArrow={hasArrow}
      {...tooltipProps}
    >
      <Button
        ref={forwardedRef}
        colorScheme={isChecked ? 'accent' : 'base'}
        {...rest}
      >
        {children}
      </Button>
    </Tooltip>
  );
});

export default memo(IAIButton);
