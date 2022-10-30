import { Button, ButtonProps, Tooltip } from '@chakra-ui/react';

export interface IAIButtonProps extends ButtonProps {
  label: string;
  tooltip?: string;
}

/**
 * Reusable customized button component.
 */
const IAIButton = (props: IAIButtonProps) => {
  const { label, tooltip = '', ...rest } = props;
  return (
    <Tooltip label={tooltip}>
      <Button {...rest}>{label}</Button>
    </Tooltip>
  );
};

export default IAIButton;
