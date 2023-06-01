import { Checkbox, CheckboxProps, Text } from '@chakra-ui/react';
import { memo, ReactNode } from 'react';

type IAISimpleCheckboxProps = CheckboxProps & {
  label: string | ReactNode;
};

const IAISimpleCheckbox = (props: IAISimpleCheckboxProps) => {
  const { label, ...rest } = props;
  return (
    <Checkbox colorScheme="accent" {...rest}>
      <Text color="base.200" fontSize="md">
        {label}
      </Text>
    </Checkbox>
  );
};

export default memo(IAISimpleCheckbox);
