import { Box, Flex } from '@chakra-ui/react';
import ModelSelect from 'features/system/components/ModelSelect';
import { memo } from 'react';
import ParamScheduler from './ParamScheduler';

const ParamSchedulerAndModel = () => {
  return (
    <Flex gap={3} w="full">
      <Box w="25rem">
        <ParamScheduler />
      </Box>
      <Box w="full">
        <ModelSelect />
      </Box>
    </Flex>
  );
};

export default memo(ParamSchedulerAndModel);
