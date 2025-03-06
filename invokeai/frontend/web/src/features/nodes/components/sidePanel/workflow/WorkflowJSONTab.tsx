import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { EMPTY_OBJECT } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import type { NodesState, WorkflowsState } from 'features/nodes/store/types';
import { selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { buildWorkflowFast } from 'features/nodes/util/workflow/buildWorkflow';
import { debounce } from 'lodash-es';
import { atom, computed } from 'nanostores';
import { memo, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

const $maybePreviewWorkflow = atom<WorkflowV3 | null>(null);
const $previewWorkflow = computed(
  $maybePreviewWorkflow,
  (maybePreviewWorkflow) => maybePreviewWorkflow ?? EMPTY_OBJECT
);

const debouncedBuildPreviewWorkflow = debounce(
  (nodes: NodesState['nodes'], edges: NodesState['edges'], workflow: WorkflowsState) => {
    $maybePreviewWorkflow.set(buildWorkflowFast({ nodes, edges, workflow }));
  },
  300
);

const IsolatedWorkflowBuilderWatcher = memo(() => {
  const { nodes, edges } = useAppSelector(selectNodesSlice);
  const workflow = useAppSelector(selectWorkflowSlice);

  useEffect(() => {
    debouncedBuildPreviewWorkflow(nodes, edges, workflow);
  }, [edges, nodes, workflow]);

  return null;
});
IsolatedWorkflowBuilderWatcher.displayName = 'IsolatedWorkflowBuilderWatcher';

const WorkflowJSONTab = () => {
  const previewWorkflow = useStore($previewWorkflow);
  const { t } = useTranslation();

  return (
    <Flex flexDir="column" alignItems="flex-start" gap={2} h="full">
      <DataViewer data={previewWorkflow} label={t('nodes.workflow')} bg="base.850" color="base.200" />
      <IsolatedWorkflowBuilderWatcher />
    </Flex>
  );
};

export default memo(WorkflowJSONTab);
