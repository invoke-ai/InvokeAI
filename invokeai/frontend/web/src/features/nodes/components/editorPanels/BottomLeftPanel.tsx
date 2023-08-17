import { memo } from 'react';
import { Panel } from 'reactflow';
import ViewportControls from '../ViewportControls';
import NodeOpacitySlider from '../NodeOpacitySlider';
import { Flex } from '@chakra-ui/react';

const BottomLeftPanel = () => (
  <Panel position="bottom-left">
    <Flex sx={{ gap: 2 }}>
      <ViewportControls />
      <NodeOpacitySlider />
    </Flex>
  </Panel>
);

export default memo(BottomLeftPanel);
