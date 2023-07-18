import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';
import { Panel } from 'reactflow';
import FieldTypeLegend from '../FieldTypeLegend';
import NodeGraphOverlay from '../NodeGraphOverlay';

const TopRightPanel = () => {
  const shouldShowGraphOverlay = useAppSelector(
    (state: RootState) => state.nodes.shouldShowGraphOverlay
  );
  const shouldShowFieldTypeLegend = useAppSelector(
    (state: RootState) => state.nodes.shouldShowFieldTypeLegend
  );

  return (
    <Panel position="top-right">
      {shouldShowFieldTypeLegend && <FieldTypeLegend />}
      {shouldShowGraphOverlay && <NodeGraphOverlay />}
    </Panel>
  );
};

export default memo(TopRightPanel);
