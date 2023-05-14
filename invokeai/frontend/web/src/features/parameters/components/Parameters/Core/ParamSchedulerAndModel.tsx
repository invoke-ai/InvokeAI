import { Box, Flex } from '@chakra-ui/react';
import { memo } from 'react';
import ParamSampler from './ParamSampler';
import ModelSelect from 'features/system/components/ModelSelect';

const ParamSchedulerAndModel = () => {
  return (
    <Flex gap={3} w="full">
      <Box w="16rem">
        <ParamSampler />
      </Box>
      <Box w="full">
        <ModelSelect />
      </Box>
    </Flex>
  );
};

export default memo(ParamSchedulerAndModel);
