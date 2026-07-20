import { useMountEffect } from '@platform/react/useMountEffect';
import { useEffectEvent } from 'react';

import type { WorkflowFlowInstance } from './flowInstanceStore';

import { clearNodeSelectionRequest, workflowSelectionStore } from './selectionStore';

interface WorkflowSelectionRequestRuntimeProps {
  flowInstance: WorkflowFlowInstance;
  reduceMotion: boolean;
  selectNodes: (nodeIds: string[]) => void;
}

/** Applies outside selection requests while a workflow flow instance is mounted. */
export const WorkflowSelectionRequestRuntime = ({
  flowInstance,
  reduceMotion,
  selectNodes,
}: WorkflowSelectionRequestRuntimeProps) => {
  const applyRequestedSelection = useEffectEvent(() => {
    const selectionRequest = workflowSelectionStore.getSnapshot().selectionRequest;

    if (!selectionRequest) {
      return;
    }

    selectNodes(selectionRequest.nodeIds);
    void flowInstance.fitView({
      duration: reduceMotion ? 0 : 300,
      maxZoom: 1.25,
      nodes: selectionRequest.nodeIds.map((id) => ({ id })),
    });
    clearNodeSelectionRequest();
  });

  /* eslint-disable react-hooks/rules-of-hooks -- useMountEffect is the repository's explicit useEffect wrapper */
  useMountEffect(() => {
    const pendingRequestTimer = window.setTimeout(applyRequestedSelection, 0);
    const unsubscribe = workflowSelectionStore.subscribe(applyRequestedSelection);

    return () => {
      window.clearTimeout(pendingRequestTimer);
      unsubscribe();
    };
  });
  /* eslint-enable react-hooks/rules-of-hooks */

  return null;
};
