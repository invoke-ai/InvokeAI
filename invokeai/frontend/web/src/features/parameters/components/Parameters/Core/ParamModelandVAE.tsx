import { Box, Flex } from '@chakra-ui/react';
import ModelSelect from 'features/system/components/ModelSelect';
import VAESelect from 'features/system/components/VAESelect';
import { memo } from 'react';

const ParamModelandVAE = () => {
  return (
    <Flex gap={3} w="full">
      <Box w="full">
        <ModelSelect />
      </Box>
      <Box w="full">
        <VAESelect />
      </Box>
    </Flex>
  );
};

export default memo(ParamModelandVAE);
