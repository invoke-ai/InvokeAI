import { Checkbox, CheckboxProps } from '@chakra-ui/react';
import type { ReactNode } from 'react';

type IAICheckboxProps = CheckboxProps & {
  label: string | ReactNode;
  styleClass?: string;
};

const IAICheckbox = (props: IAICheckboxProps) => {
  const { label, styleClass, ...rest } = props;
  return (
    <Checkbox className={`invokeai__checkbox ${styleClass}`} {...rest}>
      {label}
    </Checkbox>
  );
};

export default IAICheckbox;
