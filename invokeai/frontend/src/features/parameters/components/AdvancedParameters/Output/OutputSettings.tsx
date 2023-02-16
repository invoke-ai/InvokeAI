import { Flex } from '@chakra-ui/react';
import HiresSettings from './HiresSettings';
import SeamlessSettings from './SeamlessSettings';

const OutputSettings = () => {
  return (
    <Flex gap={2} direction="column">
      <SeamlessSettings />
      <HiresSettings />
    </Flex>
  );
};

export default OutputSettings;
