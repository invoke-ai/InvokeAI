import { Checkbox, CheckboxProps } from '@chakra-ui/react';

type IAICheckboxProps = CheckboxProps & {
  label: string;
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
