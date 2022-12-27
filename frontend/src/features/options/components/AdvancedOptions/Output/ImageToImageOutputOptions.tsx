import { Flex } from '@chakra-ui/react';
import SeamlessOptions from './SeamlessOptions';

const ImageToImageOutputOptions = () => {
  return (
    <Flex gap={2} direction={'column'}>
      <SeamlessOptions />
    </Flex>
  );
};

export default ImageToImageOutputOptions;
