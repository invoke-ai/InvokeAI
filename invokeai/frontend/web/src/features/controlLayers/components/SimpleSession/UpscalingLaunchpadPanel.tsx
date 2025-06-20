import { Flex, Heading } from '@invoke-ai/ui-library';
import { memo } from 'react';

export const UpscalingLaunchpadPanel = memo(() => {
  return (
    <Flex flexDir="column" h="full" w="full" alignItems="center" gap={2}>
      <Flex flexDir="column" w="full" gap={4} px={14} maxW={768} pt="20vh">
        <Heading mb={4}>Upscale and add detail.</Heading>
      </Flex>
    </Flex>
  );
});
UpscalingLaunchpadPanel.displayName = 'UpscalingLaunchpadPanel';
