import {
  IconButtonProps,
  IconButton,
  Tooltip,
  TooltipProps,
  ResponsiveValue,
  ThemingProps,
  isChakraTheme,
} from '@chakra-ui/react';
import { Variant } from 'framer-motion';

interface Props extends IconButtonProps {
  styleClass?: string;
  tooltip?: string;
  tooltipProps?: Omit<TooltipProps, 'children'>;
  asCheckbox?: boolean;
  isChecked?: boolean;
}

const IAIIconButton = (props: Props) => {
  const {
    tooltip = '',
    styleClass,
    tooltipProps,
    asCheckbox,
    isChecked,
    ...rest
  } = props;

  return (
    <Tooltip label={tooltip} hasArrow {...tooltipProps}>
      <IconButton
        className={`invokeai__icon-button ${styleClass}`}
        data-as-checkbox={asCheckbox}
        data-selected={isChecked !== undefined ? isChecked : undefined}
        style={props.onClick ? { cursor: 'pointer' } : {}}
        {...rest}
      />
    </Tooltip>
  );
};

export default IAIIconButton;
