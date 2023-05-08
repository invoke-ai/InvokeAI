import { VStack } from '@chakra-ui/react';
import ParamSeamlessToggle from './Seamless/ParamSeamlessToggle';
// import ParamSeamlessAxes from '../../Parameters/Seamless/ParamSeamlessAxes';
import { ParamHiresToggle } from './Hires/ParamHiresToggle';
import { ParamHiresStrength } from './Hires/ParamHiresStrength';

const OtherSettings = () => {
  return (
    <VStack gap={2} alignItems="stretch">
      <ParamSeamlessToggle />
      {/* <ParamSeamlessAxes /> */}
      <ParamHiresToggle />
      <ParamHiresStrength />
    </VStack>
  );
};

export default OtherSettings;
