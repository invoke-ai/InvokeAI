import { Flex } from '@chakra-ui/react';
import SeamlessSettings from './SeamlessSettings';

const ImageToImageOutputSettings = () => {
  return (
    <Flex gap={2} direction="column">
      <SeamlessSettings />
    </Flex>
  );
};

export default ImageToImageOutputSettings;
