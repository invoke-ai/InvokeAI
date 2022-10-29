import {
  IconButtonProps,
  IconButton,
  Tooltip,
  PlacementWithLogical,
} from '@chakra-ui/react';

interface Props extends IconButtonProps {
  tooltip?: string;
  tooltipPlacement?: PlacementWithLogical | undefined;
  styleClass?: string;
}

/**
 * Reusable customized button component. Originally was more customized - now probably unecessary.
 */
const IAIIconButton = (props: Props) => {
  const {
    tooltip = '',
    tooltipPlacement = 'bottom',
    styleClass,
    onClick,
    cursor,
    ...rest
  } = props;

  return (
    <Tooltip label={tooltip} hasArrow placement={tooltipPlacement}>
      <IconButton
        className={`icon-button ${styleClass}`}
        {...rest}
        cursor={cursor ? cursor : onClick ? 'pointer' : 'unset'}
        onClick={onClick}
      />
    </Tooltip>
  );
};

export default IAIIconButton;
