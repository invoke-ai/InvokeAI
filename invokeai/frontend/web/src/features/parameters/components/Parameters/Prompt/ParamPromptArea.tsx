import { Flex } from '@chakra-ui/react';
import ParamNegativeConditioning from 'features/parameters/components/Parameters/Core/ParamNegativeConditioning';
import ParamPositiveConditioning from 'features/parameters/components/Parameters/Core/ParamPositiveConditioning';

export default function ParamPromptArea() {
  return (
    <Flex
      sx={{
        flexDirection: 'column',
        gap: 2,
        p: 2,
        borderRadius: 4,
        bg: 'base.100',
        _dark: { bg: 'base.850' },
      }}
    >
      <ParamPositiveConditioning />
      <ParamNegativeConditioning />
    </Flex>
  );
}
