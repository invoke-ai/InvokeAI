import { Box, Flex } from '@chakra-ui/react';
import { memo } from 'react';
import ModelSelect from 'features/system/components/ModelSelect';
import ParamScheduler from './ParamScheduler';

const ParamSchedulerAndModel = () => {
  return (
    <Flex gap={3} w="full">
      <Box w="16rem">
        <ParamScheduler />
      </Box>
      <Box w="full">
        <ModelSelect />
      </Box>
    </Flex>
  );
};

export default memo(ParamSchedulerAndModel);
