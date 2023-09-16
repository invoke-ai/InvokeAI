import {
  forwardRef,
  IconButton,
  IconButtonProps,
  Tooltip,
  TooltipProps,
} from '@chakra-ui/react';
import { memo } from 'react';

export type IAIIconButtonProps = IconButtonProps & {
  role?: string;
  tooltip?: TooltipProps['label'];
  tooltipProps?: Omit<TooltipProps, 'children' | 'label'>;
  isChecked?: boolean;
};

const IAIIconButton = forwardRef((props: IAIIconButtonProps, forwardedRef) => {
  const { role, tooltip = '', tooltipProps, isChecked, ...rest } = props;

  return (
    <Tooltip
      label={tooltip}
      hasArrow
      {...tooltipProps}
      {...(tooltipProps?.placement
        ? { placement: tooltipProps.placement }
        : { placement: 'top' })}
    >
      <IconButton
        ref={forwardedRef}
        role={role}
        colorScheme={isChecked ? 'accent' : 'base'}
        {...rest}
      />
    </Tooltip>
  );
});

IAIIconButton.displayName = 'IAIIconButton';
export default memo(IAIIconButton);
