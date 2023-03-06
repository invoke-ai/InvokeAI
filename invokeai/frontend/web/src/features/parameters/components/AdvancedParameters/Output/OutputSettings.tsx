import { VStack } from '@chakra-ui/react';
import { HiresStrength, HiresToggle } from './HiresSettings';
import SeamlessSettings from './SeamlessSettings';

const OutputSettings = () => {
  return (
    <VStack gap={2} alignItems="stretch">
      <SeamlessSettings />
      <HiresToggle />
      <HiresStrength />
    </VStack>
  );
};

export default OutputSettings;
