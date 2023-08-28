import { Flex } from '@chakra-ui/react';
import { memo } from 'react';
import NodeOpacitySlider from './NodeOpacitySlider';
import ViewportControls from './ViewportControls';

const BottomLeftPanel = () => (
  <Flex sx={{ gap: 2, position: 'absolute', bottom: 2, insetInlineStart: 2 }}>
    <ViewportControls />
    <NodeOpacitySlider />
  </Flex>
);

export default memo(BottomLeftPanel);
