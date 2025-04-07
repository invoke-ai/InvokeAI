import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { EMPTY_OBJECT } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { getInitialWorkflow } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice, selectWorkflowId } from 'features/nodes/store/selectors';
import type { NodesState } from 'features/nodes/store/types';
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

const debouncedBuildPreviewWorkflow = debounce((nodesState: NodesState) => {
  $maybePreviewWorkflow.set(buildWorkflowFast(nodesState));
}, 300);

export const useWorkflowBuilderWatcher = () => {
  useAssertSingleton('useWorkflowBuilderWatcher');
  const nodesState = useAppSelector(selectNodesSlice);

  useEffect(() => {
    debouncedBuildPreviewWorkflow(nodesState);
  }, [nodesState]);
};

const queryOptions = {
  selectFromResult: ({ currentData }) => {
    if (!currentData) {
      return { serverWorkflowHash: null };
    }
    const { is_published: _is_published, ...serverWorkflow } = currentData.workflow;
    return {
      serverWorkflowHash: stableHash(serverWorkflow),
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

const initialWorkflowHash = stableHash({ ...getInitialWorkflow(), nodes: [], edges: [] });

export const useIsWorkflowUntouched = () => {
  const previewWorkflowHash = useStore($previewWorkflowHash);

  const isWorkflowUntouched = useMemo(() => {
    return previewWorkflowHash === initialWorkflowHash;
  }, [previewWorkflowHash]);

  return isWorkflowUntouched;
};
