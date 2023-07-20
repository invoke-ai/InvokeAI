import {
  FormControl,
  FormControlProps,
  FormLabel,
  FormLabelProps,
  Switch,
  SwitchProps,
  Tooltip,
} from '@chakra-ui/react';
import { memo } from 'react';

export interface IAISwitchProps extends SwitchProps {
  label?: string;
  width?: string | number;
  formControlProps?: FormControlProps;
  formLabelProps?: FormLabelProps;
  tooltip?: string;
}

/**
 * Customized Chakra FormControl + Switch multi-part component.
 */
const IAISwitch = (props: IAISwitchProps) => {
  const {
    label,
    isDisabled = false,
    width = 'auto',
    formControlProps,
    formLabelProps,
    tooltip,
    ...rest
  } = props;
  return (
    <Tooltip label={tooltip} hasArrow placement="top" isDisabled={!tooltip}>
      <FormControl
        isDisabled={isDisabled}
        width={width}
        display="flex"
        alignItems="center"
        {...formControlProps}
      >
        {label && (
          <FormLabel
            my={1}
            flexGrow={1}
            sx={{
              cursor: isDisabled ? 'not-allowed' : 'pointer',
              ...formLabelProps?.sx,
              pe: 4,
            }}
            {...formLabelProps}
          >
            {label}
          </FormLabel>
        )}
        <Switch {...rest} />
      </FormControl>
    </Tooltip>
  );
};

export default memo(IAISwitch);
