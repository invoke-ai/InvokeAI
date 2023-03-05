import {
  FormControl,
  FormControlProps,
  FormLabel,
  FormLabelProps,
  Switch,
  SwitchProps,
} from '@chakra-ui/react';

interface Props extends SwitchProps {
  label?: string;
  width?: string | number;
  formControlProps?: FormControlProps;
  formLabelProps?: FormLabelProps;
}

/**
 * Customized Chakra FormControl + Switch multi-part component.
 */
const IAISwitch = (props: Props) => {
  const {
    label,
    isDisabled = false,
    width = 'auto',
    formControlProps,
    formLabelProps,
    ...rest
  } = props;
  return (
    <FormControl
      isDisabled={isDisabled}
      width={width}
      display="flex"
      gap={4}
      alignItems="center"
      justifyContent="space-between"
      {...formControlProps}
    >
      <FormLabel my={1} {...formLabelProps}>
        {label}
      </FormLabel>
      <Switch {...rest} />
    </FormControl>
  );
};

export default IAISwitch;
