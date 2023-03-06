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
      display="flex"
      columnGap="1rem"
      alignItems="center"
      justifyContent="space-between"
      {...formControlProps}
    >
      <FormLabel
        className="invokeai__switch-form-label"
        whiteSpace="nowrap"
        marginRight={0}
        marginTop={0.5}
        marginBottom={0.5}
        fontSize="sm"
        fontWeight="bold"
        width="auto"
        {...formLabelProps}
      >
        {label}
      </FormLabel>
      <Switch className="invokeai__switch-root" {...rest} />
    </FormControl>
  );
};

export default IAISwitch;
