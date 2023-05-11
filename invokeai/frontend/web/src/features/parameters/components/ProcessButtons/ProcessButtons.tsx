import { Flex } from '@chakra-ui/react';
import CancelButton from './CancelButton';
import InvokeButton from './InvokeButton';

/**
 * Buttons to start and cancel image generation.
 */
const ProcessButtons = () => {
  return (
    <Flex gap={2}>
      <InvokeButton />
      {/* {activeTabName === 'img2img' && <LoopbackButton />} */}
      <CancelButton />
    </Flex>
  );
};

export default ProcessButtons;
