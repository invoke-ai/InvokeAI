import { Checkbox, CheckboxProps, Text, useColorMode } from '@chakra-ui/react';
import { memo, ReactElement } from 'react';
import { mode } from 'theme/util/mode';

type IAISimpleCheckboxProps = CheckboxProps & {
  label: string | ReactElement;
};

const IAISimpleCheckbox = (props: IAISimpleCheckboxProps) => {
  const { label, ...rest } = props;
  const { colorMode } = useColorMode();
  return (
    <Checkbox colorScheme="accent" {...rest}>
      <Text
        sx={{
          fontSize: 'sm',
          color: mode('base.800', 'base.200')(colorMode),
        }}
      >
        {label}
      </Text>
    </Checkbox>
  );
};

export default memo(IAISimpleCheckbox);
