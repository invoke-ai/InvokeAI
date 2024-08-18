import { Box } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useScopeOnFocus } from 'common/hooks/interactionScopes';
import NodeEditor from 'features/nodes/components/NodeEditor';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo, useRef } from 'react';
import { ReactFlowProvider } from 'reactflow';

const NodesTab = () => {
  const mode = useAppSelector((s) => s.workflow.mode);
  const activeTabName = useAppSelector(activeTabNameSelector);
  const ref = useRef<HTMLDivElement>(null);
  useScopeOnFocus('workflows', ref);

  return (
    <Box
      display={activeTabName === 'workflows' ? undefined : 'none'}
      hidden={activeTabName !== 'workflows'}
      ref={ref}
      layerStyle="first"
      position="relative"
      w="full"
      h="full"
      p={2}
      borderRadius="base"
    >
      {mode === 'edit' && (
        <ReactFlowProvider>
          <NodeEditor />
        </ReactFlowProvider>
      )}
    </Box>
  );
};

export default memo(NodesTab);
