import { Box, Flex } from '@chakra-ui/react';
import ModelSelect from 'features/system/components/ModelSelect';
import VAESelect from 'features/system/components/VAESelect';
import { memo } from 'react';
import { useFeatureStatus } from '../../../../system/hooks/useFeatureStatus';
import ParamScheduler from './ParamScheduler';

const ParamModelandVAEandScheduler = () => {
  const isVaeEnabled = useFeatureStatus('vae').isFeatureEnabled;

  return (
    <Flex gap={3} w="full" flexWrap={isVaeEnabled ? 'wrap' : 'nowrap'}>
      <Box w="full">
        <ModelSelect />
      </Box>
      <Flex gap={3} w="full">
        {isVaeEnabled && (
          <Box w="full">
            <VAESelect />
          </Box>
        )}
        <Box w="full">
          <ParamScheduler />
        </Box>
      </Flex>
    </Flex>
  );
};

export default memo(ParamModelandVAEandScheduler);
