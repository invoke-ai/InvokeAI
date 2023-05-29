import { Flex } from '@chakra-ui/react';
import ControlNetTab1 from './ControlNetTab1';
import ControlNetTab2 from './ControlNetTab2';
import ControlNetTab3 from './ControlNetTab3';

export default function ControlNet() {
  return (
    <Flex sx={{ minWidth: '30%', flexDirection: 'column', rowGap: 2 }}>
      <ControlNetTab1 />
      <ControlNetTab2 />
      <ControlNetTab3 />
    </Flex>
  );
}
