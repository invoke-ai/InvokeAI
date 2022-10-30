import { Button, ButtonProps, Tooltip } from '@chakra-ui/react';

export interface IAIButtonProps extends ButtonProps {
  label: string;
  tooltip?: string;
  styleClass?: string;
}

/**
 * Reusable customized button component.
 */
const IAIButton = (props: IAIButtonProps) => {
  const { label, tooltip = '', styleClass, ...rest } = props;
  return (
    <Tooltip label={tooltip}>
      <Button className={styleClass ? styleClass : ''} {...rest}>
        {label}
      </Button>
    </Tooltip>
  );
};

export default IAIButton;
