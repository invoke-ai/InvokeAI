import {
  Checkbox,
  CheckboxProps,
  FormControl,
  FormControlProps,
  FormLabel,
} from '@chakra-ui/react';
import { memo, ReactNode } from 'react';

type IAIFullCheckboxProps = CheckboxProps & {
  label: string | ReactNode;
  formControlProps?: FormControlProps;
};

const IAIFullCheckbox = (props: IAIFullCheckboxProps) => {
  const { label, formControlProps, ...rest } = props;
  return (
    <FormControl {...formControlProps}>
      <FormLabel>{label}</FormLabel>
      <Checkbox colorScheme="accent" {...rest} />
    </FormControl>
  );
};

export default memo(IAIFullCheckbox);
