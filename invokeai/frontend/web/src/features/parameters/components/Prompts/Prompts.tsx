import { Flex } from '@chakra-ui/react';
import { ParamNegativePrompt } from 'features/parameters/components/Core/ParamNegativePrompt';
import { ParamPositivePrompt } from 'features/parameters/components/Core/ParamPositivePrompt';

export const Prompts = () => {
  return (
    <Flex flexDir="column" gap={2}>
      <ParamPositivePrompt />
      <ParamNegativePrompt />
    </Flex>
  );
};
