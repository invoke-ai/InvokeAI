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
  styleClass?: string;
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
    styleClass,
    ...rest
  } = props;
  return (
    <FormControl
      isDisabled={isDisabled}
      width={width}
      className={`invokeai__switch-form-control ${styleClass}`}
      {...formControlProps}
    >
      <FormLabel
        className="invokeai__switch-form-label"
        whiteSpace="nowrap"
        {...formLabelProps}
      >
        {label}
        <Switch className="invokeai__switch-root" {...rest} />
      </FormLabel>
    </FormControl>
  );
};

export default IAISwitch;
