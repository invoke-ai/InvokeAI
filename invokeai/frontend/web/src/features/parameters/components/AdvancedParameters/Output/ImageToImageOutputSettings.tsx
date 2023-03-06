import { VStack } from '@chakra-ui/react';
import SeamlessSettings from './SeamlessSettings';

const ImageToImageOutputSettings = () => {
  return (
    <VStack gap={2} alignItems="stretch">
      <SeamlessSettings />
    </VStack>
  );
};

export default ImageToImageOutputSettings;
