import { Box, Flex } from '@chakra-ui/react';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';
import ParamMainModelSelect from 'features/parameters/components/Parameters/MainModel/ParamMainModelSelect';
import ParamVAEModelSelect from 'features/parameters/components/Parameters/VAEModel/ParamVAEModelSelect';
import ParamScheduler from './ParamScheduler';
import ParamVAEPrecision from 'features/parameters/components/Parameters/VAEModel/ParamVAEPrecision';

const ParamModelandVAEandScheduler = () => {
  const isVaeEnabled = useFeatureStatus('vae').isFeatureEnabled;

  return (
    <Flex gap={3} w="full" flexWrap={isVaeEnabled ? 'wrap' : 'nowrap'}>
      <Box w="full">
        <ParamMainModelSelect />
      </Box>
      <Box w="full">
        <ParamScheduler />
      </Box>
      {isVaeEnabled && (
        <Flex w="full" gap={3}>
          <ParamVAEModelSelect />
          <ParamVAEPrecision />
        </Flex>
      )}
    </Flex>
  );
};

export default memo(ParamModelandVAEandScheduler);
