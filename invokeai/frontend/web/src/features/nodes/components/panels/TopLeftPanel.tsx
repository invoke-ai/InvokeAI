import { memo } from 'react';
import { Panel } from 'reactflow';
import NodeSearch from '../search/NodeSearch';

const TopLeftPanel = () => (
  <Panel position="top-left">
    <NodeSearch />
  </Panel>
);

export default memo(TopLeftPanel);
