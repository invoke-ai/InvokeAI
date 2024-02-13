import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import type { BuildWorkflowArg } from 'features/nodes/util/workflow/buildWorkflow';
import { buildWorkflowFast } from 'features/nodes/util/workflow/buildWorkflow';
import { debounce } from 'lodash-es';
import { atom } from 'nanostores';
import { useEffect } from 'react';

export const $builtWorkflow = atom<WorkflowV3 | null>(null);

const debouncedBuildWorkflow = debounce((arg: BuildWorkflowArg) => {
  $builtWorkflow.set(buildWorkflowFast(arg));
}, 300);

const selectWorkflowSlices = createSelector(selectNodesSlice, selectWorkflowSlice, (nodes, workflow) => ({
  nodes: nodes.nodes,
  edges: nodes.edges,
  workflow,
}));

export const useWorkflowWatcher = () => {
  const buildWorkflowArg = useAppSelector(selectWorkflowSlices);

  useEffect(() => {
    debouncedBuildWorkflow(buildWorkflowArg);
  }, [buildWorkflowArg]);
};
