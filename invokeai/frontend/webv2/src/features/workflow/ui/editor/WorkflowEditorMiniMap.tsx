import type { Node } from '@xyflow/react';

import {
  FLOW_MINIMAP_ERROR_CLASS,
  FLOW_MINIMAP_RUNNING_CLASS,
  FlowMiniMap,
} from '@features/workflow/ui/graph-preview/FlowMiniMap';
import { useWorkflowUi } from '@features/workflow/ui/WorkflowUiContext';
import { useCallback, useMemo, useSyncExternalStore } from 'react';

import type { InvocationFlowNode as InvocationFlowNodeType } from './flowAdapters';

import { isInvocationFlowNodeInvalid } from './InvocationFlowNode';

/**
 * The run state of this editor's nodes that the minimap shows, as `id:status` lines for the running and failed ones.
 * A string snapshot, so progress ticks and unrelated project changes do not render the minimap.
 */
const useMinimapRunStates = (nodeIds: readonly string[]): Map<string, 'failed' | 'running'> => {
  const { nodeExecution, project } = useWorkflowUi();
  const subscribe = useCallback(
    (listener: () => void) => {
      const unsubscribers = [
        ...nodeIds.map((nodeId) => nodeExecution.subscribe(nodeId, listener)),
        nodeExecution.subscribeOrigin(listener),
        project.subscribe(listener),
      ];

      return () => unsubscribers.forEach((unsubscribe) => unsubscribe());
    },
    [nodeExecution, nodeIds, project]
  );
  const getSnapshot = useCallback(() => {
    const origin = nodeExecution.getOrigin();
    const snapshot = project.getSnapshot();

    if (!origin || origin.projectId !== snapshot.id || origin.workflowId !== snapshot.activeWorkflowId) {
      return '';
    }

    return nodeIds
      .flatMap((nodeId) => {
        const status = nodeExecution.get(nodeId)?.status;

        return status === 'running' || status === 'failed' ? [`${nodeId}:${status}`] : [];
      })
      .join('\n');
  }, [nodeExecution, nodeIds, project]);
  const states = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);

  return useMemo(
    () =>
      new Map(
        states
          .split('\n')
          .filter(Boolean)
          .map((line) => {
            const separator = line.lastIndexOf(':');

            return [line.slice(0, separator), line.slice(separator + 1) as 'failed' | 'running'];
          })
      ),
    [states]
  );
};

/**
 * The editor's minimap: invalid and failed nodes in the error tone, the running node in the accent tone. `nodeIds` are
 * the document's invocation nodes, which the editor derives only when the node set changes, not on every drag frame.
 */
export const WorkflowEditorMiniMap = ({ nodeIds }: { nodeIds: readonly string[] }) => {
  const runStates = useMinimapRunStates(nodeIds);
  const nodeClassName = useCallback(
    (node: Node) => {
      if (node.type !== 'invocation') {
        return '';
      }

      const runState = runStates.get(node.id);

      return runState === 'running'
        ? FLOW_MINIMAP_RUNNING_CLASS
        : runState === 'failed' || isInvocationFlowNodeInvalid((node as InvocationFlowNodeType).data)
          ? FLOW_MINIMAP_ERROR_CLASS
          : '';
    },
    [runStates]
  );

  return <FlowMiniMap nodeClassName={nodeClassName} />;
};
