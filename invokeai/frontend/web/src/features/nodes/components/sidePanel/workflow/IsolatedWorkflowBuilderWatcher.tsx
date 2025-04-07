import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { EMPTY_OBJECT } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import type { NodesState, WorkflowsState } from 'features/nodes/store/types';
import { getBlankWorkflow, selectWorkflowId, selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { buildWorkflowFast } from 'features/nodes/util/workflow/buildWorkflow';
import { debounce } from 'lodash-es';
import { atom, computed } from 'nanostores';
import { useEffect, useMemo } from 'react';
import { useGetWorkflowQuery } from 'services/api/endpoints/workflows';
import stableHash from 'stable-hash';

const $maybePreviewWorkflow = atom<WorkflowV3 | null>(null);
export const $previewWorkflow = computed(
  $maybePreviewWorkflow,
  (maybePreviewWorkflow) => maybePreviewWorkflow ?? EMPTY_OBJECT
);
const $previewWorkflowHash = computed($maybePreviewWorkflow, (maybePreviewWorkflow) => {
  if (maybePreviewWorkflow) {
    return stableHash(maybePreviewWorkflow);
  }
  return null;
});

const debouncedBuildPreviewWorkflow = debounce(
  (nodes: NodesState['nodes'], edges: NodesState['edges'], workflow: WorkflowsState) => {
    $maybePreviewWorkflow.set(buildWorkflowFast({ nodes, edges, workflow }));
  },
  300
);

export const useWorkflowBuilderWatcher = () => {
  useAssertSingleton('useWorkflowBuilderWatcher');
  const { nodes, edges } = useAppSelector(selectNodesSlice);
  const workflow = useAppSelector(selectWorkflowSlice);

  useEffect(() => {
    debouncedBuildPreviewWorkflow(nodes, edges, workflow);
  }, [edges, nodes, workflow]);
};

const queryOptions = {
  selectFromResult: ({ currentData }) => {
    if (!currentData) {
      return { serverWorkflowHash: null };
    }
    return {
      serverWorkflowHash: stableHash(currentData.workflow),
    };
  },
} satisfies Parameters<typeof useGetWorkflowQuery>[1];

export const useDoesWorkflowHaveUnsavedChanges = () => {
  const workflowId = useAppSelector(selectWorkflowId);
  const previewWorkflowHash = useStore($previewWorkflowHash);
  const { serverWorkflowHash } = useGetWorkflowQuery(workflowId ?? skipToken, queryOptions);

  const doesWorkflowHaveUnsavedChanges = useMemo(() => {
    if (serverWorkflowHash === null) {
      // If the hash is null, it means the workflow doesn't exist in the database
      return true;
    }
    return previewWorkflowHash !== serverWorkflowHash;
  }, [previewWorkflowHash, serverWorkflowHash]);

  return doesWorkflowHaveUnsavedChanges;
};

const initialWorkflowHash = stableHash({ ...getBlankWorkflow(), nodes: [], edges: [] });

export const useIsWorkflowUntouched = () => {
  const previewWorkflowHash = useStore($previewWorkflowHash);

  const isWorkflowUntouched = useMemo(() => {
    return previewWorkflowHash === initialWorkflowHash;
  }, [previewWorkflowHash]);

  return isWorkflowUntouched;
};
