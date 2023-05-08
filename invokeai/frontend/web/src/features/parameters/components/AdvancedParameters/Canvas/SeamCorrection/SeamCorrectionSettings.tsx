import { VStack } from '@chakra-ui/react';
import ParamSeamBlur from './ParamSeamBlur';
import ParamSeamSize from './ParamSeamSize';
import ParamSeamSteps from './ParamSeamSteps';
import ParamSeamStrength from './ParamSeamStrength';

const SeamCorrectionSettings = () => {
  return (
    <VStack gap={2} alignItems="stretch">
      <ParamSeamSize />
      <ParamSeamBlur />
      <ParamSeamStrength />
      <ParamSeamSteps />
    </VStack>
  );
};

export default SeamCorrectionSettings;
