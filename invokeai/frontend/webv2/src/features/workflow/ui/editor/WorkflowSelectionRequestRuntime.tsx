import type { InvocationTemplates } from '@features/workflow/contracts';
import type { InternalNode, Rect, Transform } from '@xyflow/react';

import { getInvocationTemplatesSnapshot, subscribeInvocationTemplates } from '@features/workflow/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useStoreApi } from '@xyflow/react';
import { useEffectEvent, useRef } from 'react';

import type { WorkflowFlowEdge, WorkflowFlowNode } from './flowAdapters';

import {
  clearWorkflowFitViewRequest,
  workflowFitViewRequestStore,
  type WorkflowFitViewTarget,
  type WorkflowFlowInstance,
} from './flowInstanceStore';
import { clearNodeSelectionRequest, workflowSelectionStore } from './selectionStore';

// A large graph renders only visible nodes, so xyflow never measures the rest; they count at a typical node size.
const UNMEASURED_NODE_SIZE = { height: 160, width: 288 };

/** A node whose template exists renders as a placeholder card until its rebuilt data carries that template. */
const isAwaitingTemplate = (internal: InternalNode<WorkflowFlowNode>, templates: InvocationTemplates): boolean => {
  const { userNode } = internal.internals;

  return (
    userNode.type === 'invocation' &&
    userNode.data.template === null &&
    templates[userNode.data.documentNode.data.type] !== undefined
  );
};

/**
 * The bounds of exactly the target nodes once the flow shows them at their positions with their templates, measured
 * unless the graph is large; null while the flow still shows another document, an older position, a placeholder
 * card, or unmeasured nodes.
 */
const getWorkflowFitBounds = (
  nodeLookup: ReadonlyMap<string, InternalNode<WorkflowFlowNode>>,
  targets: readonly WorkflowFitViewTarget[],
  { allowUnmeasured, templates }: { allowUnmeasured: boolean; templates: InvocationTemplates }
): Rect | null => {
  if (targets.length === 0 || nodeLookup.size !== targets.length) {
    return null;
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const target of targets) {
    const internal = nodeLookup.get(target.id);
    const position = internal?.internals.positionAbsolute;

    if (
      !internal ||
      position?.x !== target.position.x ||
      position.y !== target.position.y ||
      isAwaitingTemplate(internal, templates)
    ) {
      return null;
    }

    const isMeasured = Boolean(internal.measured.width && internal.measured.height);

    if (!isMeasured && !allowUnmeasured) {
      return null;
    }

    const width = isMeasured ? internal.measured.width! : UNMEASURED_NODE_SIZE.width;
    const height = isMeasured ? internal.measured.height! : UNMEASURED_NODE_SIZE.height;

    minX = Math.min(minX, position.x);
    minY = Math.min(minY, position.y);
    maxX = Math.max(maxX, position.x + width);
    maxY = Math.max(maxY, position.y + height);
  }

  return { height: maxY - minY, width: maxX - minX, x: minX, y: minY };
};

interface WorkflowSelectionRequestRuntimeProps {
  /** The document's nodes when this editor opened without a remembered viewport; fitted once they are shown. */
  fitOnMount: readonly WorkflowFitViewTarget[] | null;
  flowInstance: WorkflowFlowInstance;
  isLargeGraph: boolean;
  projectId: string;
  reduceMotion: boolean;
  selectNodes: (nodeIds: string[]) => void;
  workflowId: string;
}

/** Applies outside selection and fit-view requests while a workflow flow instance is mounted. */
export const WorkflowSelectionRequestRuntime = ({
  fitOnMount,
  flowInstance,
  isLargeGraph,
  projectId,
  reduceMotion,
  selectNodes,
  workflowId,
}: WorkflowSelectionRequestRuntimeProps) => {
  const store = useStoreApi<WorkflowFlowNode, WorkflowFlowEdge>();
  const pendingMountFit = useRef(fitOnMount);
  // The viewport the editor opened at; a move away from it before the mount fit lands is the user's, and wins.
  const mountTransform = useRef<Transform | null>(null);
  const settleFrame = useRef<number | null>(null);
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
  const getBounds = (targets: readonly WorkflowFitViewTarget[]): Rect | null => {
    const templatesSnapshot = getInvocationTemplatesSnapshot();

    // Nodes rendered before templates load are placeholder cards that grow once their template arrives.
    if (templatesSnapshot.status === 'idle' || templatesSnapshot.status === 'loading') {
      return null;
    }

    return getWorkflowFitBounds(store.getState().nodeLookup, targets, {
      allowUnmeasured: isLargeGraph,
      templates: templatesSnapshot.templates,
    });
  };
  const getRequestTargets = () => {
    const request = workflowFitViewRequestStore.getSnapshot().request;

    return request && request.scope.projectId === projectId && request.scope.workflowId === workflowId
      ? request.nodes
      : null;
  };
  // Each target is released before fitting: a zero-duration fit updates the flow store synchronously, which
  // re-enters here.
  const fitReadyTargets = useEffectEvent(() => {
    settleFrame.current = null;

    const mountTargets = pendingMountFit.current;
    const mountBounds = mountTargets ? getBounds(mountTargets) : null;

    if (mountBounds) {
      pendingMountFit.current = null;
      void flowInstance.fitBounds(mountBounds, { duration: 0 });
    }

    const requestTargets = getRequestTargets();
    const requestBounds = requestTargets ? getBounds(requestTargets) : null;

    if (requestBounds) {
      clearWorkflowFitViewRequest();
      void flowInstance.fitBounds(requestBounds, { duration: reduceMotion ? 0 : 300 });
    }
  });
  // Targets are the document's nodes; the flow shows them only after React commits the rebuilt model (and, for a
  // large graph, its full mount). Rebuilt nodes keep their previous measured size until the resize observer runs,
  // so a ready target is fitted two frames later, from bounds read again then.
  const applyRequestedFit = useEffectEvent(() => {
    const { transform } = store.getState();

    if (pendingMountFit.current?.length === 0) {
      pendingMountFit.current = null;
    }

    if (pendingMountFit.current) {
      mountTransform.current ??= transform;

      if (mountTransform.current.some((value, index) => value !== transform[index])) {
        pendingMountFit.current = null;
      }
    }

    const requestTargets = getRequestTargets();

    if (requestTargets?.length === 0) {
      clearWorkflowFitViewRequest();
    }

    const isReady =
      (pendingMountFit.current !== null && getBounds(pendingMountFit.current) !== null) ||
      (requestTargets !== null && requestTargets.length > 0 && getBounds(requestTargets) !== null);

    if (isReady && settleFrame.current === null) {
      settleFrame.current = requestAnimationFrame(() => {
        settleFrame.current = requestAnimationFrame(fitReadyTargets);
      });
    }
  });

  /* eslint-disable react-hooks/rules-of-hooks -- useMountEffect is the repository's explicit useEffect wrapper */
  useMountEffect(() => {
    const pendingRequestTimer = window.setTimeout(() => {
      applyRequestedSelection();
      applyRequestedFit();
    }, 0);
    const unsubscribeSelection = workflowSelectionStore.subscribe(applyRequestedSelection);
    const unsubscribeFitRequests = workflowFitViewRequestStore.subscribe(applyRequestedFit);
    const unsubscribeTemplates = subscribeInvocationTemplates(applyRequestedFit);
    // The flow store ticks when nodes mount, measure, or move, which is when a pending fit becomes applicable.
    const unsubscribeFlow = store.subscribe(applyRequestedFit);

    return () => {
      window.clearTimeout(pendingRequestTimer);

      if (settleFrame.current !== null) {
        cancelAnimationFrame(settleFrame.current);
      }

      unsubscribeSelection();
      unsubscribeFitRequests();
      unsubscribeTemplates();
      unsubscribeFlow();
    };
  });
  /* eslint-enable react-hooks/rules-of-hooks */

  return null;
};
