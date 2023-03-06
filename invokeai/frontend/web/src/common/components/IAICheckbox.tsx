import { Checkbox, CheckboxProps } from '@chakra-ui/react';
import type { ReactNode } from 'react';

type IAICheckboxProps = CheckboxProps & {
  label: string | ReactNode;
};

const IAICheckbox = (props: IAICheckboxProps) => {
  const { label, ...rest } = props;
  return (
    <Checkbox colorScheme="accent" {...rest}>
      {label}
    </Checkbox>
  );
};

export default IAICheckbox;
