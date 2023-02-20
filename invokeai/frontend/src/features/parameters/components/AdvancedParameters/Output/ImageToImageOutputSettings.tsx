import { Flex } from '@chakra-ui/react';
import SeamlessSettings from './SeamlessSettings';
import SymmetrySettings from './SymmetrySettings';

const ImageToImageOutputSettings = () => {
  return (
    <Flex gap={2} direction="column">
      <SeamlessSettings />
      <SymmetrySettings />
    </Flex>
  );
};

export default ImageToImageOutputSettings;
