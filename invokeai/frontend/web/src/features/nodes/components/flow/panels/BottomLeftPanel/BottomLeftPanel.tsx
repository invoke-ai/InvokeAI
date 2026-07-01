import { Flex } from '@invoke-ai/ui-library';
import { memo } from 'react';

import NodeOpacitySlider from './NodeOpacitySlider';
import ViewportControls from './ViewportControls';

const BottomLeftPanel = () => (
  <Flex gap={2} position="absolute" bottom={2} insetInlineStart={2}>
    <ViewportControls />
    <NodeOpacitySlider />
  </Flex>
);

export default memo(BottomLeftPanel);
