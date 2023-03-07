import { VStack } from '@chakra-ui/react';
import SeamBlur from './SeamBlur';
import SeamSize from './SeamSize';
import SeamSteps from './SeamSteps';
import SeamStrength from './SeamStrength';

const SeamCorrectionSettings = () => {
  return (
    <VStack gap={2} alignItems="stretch">
      <SeamSize />
      <SeamBlur />
      <SeamStrength />
      <SeamSteps />
    </VStack>
  );
};

export default SeamCorrectionSettings;
