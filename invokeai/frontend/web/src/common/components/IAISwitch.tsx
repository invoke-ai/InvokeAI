import {
  Flex,
  FormControl,
  FormControlProps,
  FormHelperText,
  FormLabel,
  FormLabelProps,
  Switch,
  SwitchProps,
  Text,
  Tooltip,
} from '@chakra-ui/react';
import { memo } from 'react';

export interface IAISwitchProps extends SwitchProps {
  label?: string;
  width?: string | number;
  formControlProps?: FormControlProps;
  formLabelProps?: FormLabelProps;
  tooltip?: string;
  helperText?: string;
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
    helperText,
    ...rest
  } = props;
  return (
    <Tooltip label={tooltip} hasArrow placement="top" isDisabled={!tooltip}>
      <FormControl
        isDisabled={isDisabled}
        width={width}
        alignItems="center"
        {...formControlProps}
      >
        <Flex sx={{ flexDir: 'column', w: 'full' }}>
          <Flex sx={{ alignItems: 'center', w: 'full' }}>
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
          </Flex>
          {helperText && (
            <FormHelperText>
              <Text variant="subtext">{helperText}</Text>
            </FormHelperText>
          )}
        </Flex>
      </FormControl>
    </Tooltip>
  );
};

export default memo(IAISwitch);
