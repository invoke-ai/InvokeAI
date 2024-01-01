import { useAppSelector } from 'app/store/storeHooks';
import type { WorkflowV2 } from 'features/nodes/types/workflow';
import type { BuildWorkflowArg } from 'features/nodes/util/workflow/buildWorkflow';
import { buildWorkflow } from 'features/nodes/util/workflow/buildWorkflow';
import { debounce } from 'lodash-es';
import { atom } from 'nanostores';
import { useEffect } from 'react';

export const $builtWorkflow = atom<WorkflowV2 | null>(null);

const debouncedBuildWorkflow = debounce((arg: BuildWorkflowArg) => {
  $builtWorkflow.set(buildWorkflow(arg));
}, 300);

export const useWorkflowWatcher = () => {
  const buildWorkflowArg = useAppSelector(({ nodes, workflow }) => ({
    nodes: nodes.nodes,
    edges: nodes.edges,
    workflow,
  }));

  useEffect(() => {
    debouncedBuildWorkflow(buildWorkflowArg);
  }, [buildWorkflowArg]);
};
