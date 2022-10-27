import {
  Flex,
  FormControl,
  FormLabel,
  Switch,
  SwitchProps,
} from '@chakra-ui/react';

interface Props extends SwitchProps {
  label?: string;
  width?: string | number;
}

/**
 * Customized Chakra FormControl + Switch multi-part component.
 */
const IAISwitch = (props: Props) => {
  const {
    label,
    isDisabled = false,
    fontSize = 'md',
    size = 'md',
    width = 'auto',
    ...rest
  } = props;
  return (
    <FormControl
      isDisabled={isDisabled}
      width={width}
      className="invokeai__switch-form-control"
    >
      <FormLabel
        className="invokeai__switch-form-label"
        fontSize={fontSize}
        whiteSpace="nowrap"
      >
        {label}
        <Switch
          className="invokeai__switch-root"
          size={size}
          // className="switch-button"
          {...rest}
        />
      </FormLabel>
    </FormControl>
  );
};

export default IAISwitch;
