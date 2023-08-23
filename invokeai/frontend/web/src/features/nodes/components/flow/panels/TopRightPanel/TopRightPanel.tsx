import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';
import { Panel } from 'reactflow';
import FieldTypeLegend from './FieldTypeLegend';
import WorkflowEditorSettings from './WorkflowEditorSettings';

const TopRightPanel = () => {
  const shouldShowFieldTypeLegend = useAppSelector(
    (state) => state.nodes.shouldShowFieldTypeLegend
  );

  return (
    <Panel position="top-right">
      <WorkflowEditorSettings />
      {shouldShowFieldTypeLegend && <FieldTypeLegend />}
    </Panel>
  );
};

export default memo(TopRightPanel);
