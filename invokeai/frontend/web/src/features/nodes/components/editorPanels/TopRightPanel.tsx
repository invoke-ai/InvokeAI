import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';
import { Panel } from 'reactflow';
import FieldTypeLegend from '../FieldTypeLegend';

const TopRightPanel = () => {
  const shouldShowFieldTypeLegend = useAppSelector(
    (state) => state.nodes.shouldShowFieldTypeLegend
  );

  return (
    <Panel position="top-right">
      {shouldShowFieldTypeLegend && <FieldTypeLegend />}
    </Panel>
  );
};

export default memo(TopRightPanel);
