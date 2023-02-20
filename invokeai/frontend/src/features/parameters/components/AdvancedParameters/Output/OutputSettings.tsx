import { Flex } from '@chakra-ui/react';
import HiresSettings from './HiresSettings';
import SeamlessSettings from './SeamlessSettings';
import SymmetrySettings from './SymmetrySettings';

const OutputSettings = () => {
  return (
    <Flex gap={2} direction="column">
      <SeamlessSettings />
      <HiresSettings />
      <SymmetrySettings />
    </Flex>
  );
};

export default OutputSettings;
