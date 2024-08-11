import { ModelManagerLeftPanel } from 'features/modelManagerV2/subpanels/ModelManagerLeftPanel';
import { ModelPane } from 'features/modelManagerV2/subpanels/ModelPane';
import ResizeHandle from 'features/ui/components/tabs/ResizeHandle';
import { memo } from 'react';
import { Panel, PanelGroup } from 'react-resizable-panels';

const ModelManagerTab = () => {
  return (
    <PanelGroup direction="horizontal">
      <Panel>
        <ModelManagerLeftPanel />
      </Panel>
      <ResizeHandle orientation="vertical" />
      <Panel>
        <ModelPane />
      </Panel>
    </PanelGroup>
  );
};

export default memo(ModelManagerTab);
