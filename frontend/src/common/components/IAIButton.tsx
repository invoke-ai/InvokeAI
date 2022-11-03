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
  styleClass?: string;
  children: ReactNode;
}

const IAIButton = forwardRef((props: IAIButtonProps, forwardedRef) => {
  const { children, tooltip = '', tooltipProps, styleClass, ...rest } = props;
  return (
    <Tooltip label={tooltip} {...tooltipProps}>
      <Button
        ref={forwardedRef}
        className={['invokeai__button', styleClass].join(' ')}
        {...rest}
      >
        {children}
      </Button>
    </Tooltip>
  );
});

export default IAIButton;
