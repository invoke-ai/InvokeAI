import { Flex } from '@chakra-ui/react';
import UpscaleDenoisingStrength from './UpscaleDenoisingStrength';
import UpscaleStrength from './UpscaleStrength';
import UpscaleScale from './UpscaleScale';

/**
 * Displays upscaling/ESRGAN options (level and strength).
 */
const UpscaleSettings = () => {
  return (
    <Flex flexDir="column" rowGap={2} minWidth="20rem">
      <UpscaleScale />
      <UpscaleDenoisingStrength />
      <UpscaleStrength />
    </Flex>
  );
};

export default UpscaleSettings;
