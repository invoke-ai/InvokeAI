import { Flex } from '@chakra-ui/react';
import { memo } from 'react';
import CancelButton from './CancelButton';
import InvokeButton from './InvokeButton';

/**
 * Buttons to start and cancel image generation.
 */
const ProcessButtons = () => {
  return (
    <Flex layerStyle="first" sx={{ gap: 2, borderRadius: 'base', p: 2 }}>
      <InvokeButton />
      <CancelButton />
    </Flex>
  );
};

export default memo(ProcessButtons);
