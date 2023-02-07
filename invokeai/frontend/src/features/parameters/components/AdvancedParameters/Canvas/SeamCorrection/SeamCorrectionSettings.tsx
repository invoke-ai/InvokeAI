import { Flex } from '@chakra-ui/react';
import SeamBlur from './SeamBlur';
import SeamSize from './SeamSize';
import SeamSteps from './SeamSteps';
import SeamStrength from './SeamStrength';

const SeamCorrectionSettings = () => {
  return (
    <Flex direction="column" gap="1rem">
      <SeamSize />
      <SeamBlur />
      <SeamStrength />
      <SeamSteps />
    </Flex>
  );
};

export default SeamCorrectionSettings;
