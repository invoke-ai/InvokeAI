import {
  forwardRef,
  IconButton,
  IconButtonProps,
  Tooltip,
  TooltipProps,
} from '@chakra-ui/react';

export type IAIIconButtonProps = IconButtonProps & {
  tooltip?: string;
  tooltipProps?: Omit<TooltipProps, 'children'>;
  isChecked?: boolean;
};

const IAIIconButton = forwardRef((props: IAIIconButtonProps, forwardedRef) => {
  const { tooltip = '', tooltipProps, isChecked, ...rest } = props;

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
        aria-checked={isChecked !== undefined ? isChecked : undefined}
        {...rest}
      />
    </Tooltip>
  );
});

export default IAIIconButton;
