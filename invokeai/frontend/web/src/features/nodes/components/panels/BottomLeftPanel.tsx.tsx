import { memo } from 'react';
import { Panel } from 'reactflow';
import ViewportControls from '../ViewportControls';

const BottomLeftPanel = () => (
  <Panel position="bottom-left">
    <ViewportControls />
  </Panel>
);

export default memo(BottomLeftPanel);
