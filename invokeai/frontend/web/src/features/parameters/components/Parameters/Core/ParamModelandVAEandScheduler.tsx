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
      <Flex gap={3} w="full">
        <Box w="full">
          <ModelSelect />
        </Box>

        {isVaeEnabled && (
          <Box w="full">
            <VAESelect />
          </Box>
        )}
      </Flex>
      <Box w="full">
        <ParamScheduler />
      </Box>
    </Flex>
  );
};

export default memo(ParamModelandVAEandScheduler);
