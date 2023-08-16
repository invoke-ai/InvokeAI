import { memo } from 'react';
import { Panel } from 'reactflow';
import AddNodeMenu from '../AddNodeMenu';

const TopLeftPanel = () => (
  <Panel position="top-left">
    <AddNodeMenu />
  </Panel>
);

export default memo(TopLeftPanel);
