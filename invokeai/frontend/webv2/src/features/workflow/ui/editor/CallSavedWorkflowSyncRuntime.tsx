import type { ProjectGraphState } from '@features/workflow/core/types';
import type { ParsedWorkflow } from '@features/workflow/core/workflowJson';
import type { WorkflowRecordDTO } from '@features/workflow/data/api';

import { onWorkflowLibraryCacheInvalidated } from '@features/workflow/data/libraryCache';
import {
  getSavedWorkflowDetailQueryStatus,
  isSavedWorkflowDetailQueryKey,
  savedWorkflowDetailQueryKey,
  savedWorkflowDetailQueryOptions,
  shouldRetrySavedWorkflowDetailAfterFailure,
  shouldFetchSavedWorkflowDetail,
} from '@features/workflow/data/savedWorkflowQueries';
import { getInvocationTemplatesSnapshot, subscribeInvocationTemplates } from '@features/workflow/data/templates';
import { useWorkflowUi } from '@features/workflow/ui/WorkflowUiContext';
import {
  CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX,
  getSavedWorkflowDynamicEdgeIdsToRemove,
  getSavedWorkflowDynamicFields,
  getSelectedSavedWorkflow,
  parseWorkflowJson,
} from '@features/workflow/utility';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useQueryClient } from '@tanstack/react-query';
import { useMemo, useRef } from 'react';

export const createSavedWorkflowDocumentParser = (
  parse: (workflow: Record<string, unknown>) => ParsedWorkflow = parseWorkflowJson
): ((workflow: unknown) => ProjectGraphState | undefined) => {
  const cache = new WeakMap<object, ProjectGraphState | null>();

  return (workflow) => {
    if (typeof workflow !== 'object' || workflow === null || Array.isArray(workflow)) {
      return undefined;
    }

    if (cache.has(workflow)) {
      return cache.get(workflow) ?? undefined;
    }

    try {
      const document = parse(workflow as Record<string, unknown>).document;
      cache.set(workflow, document);
      return document;
    } catch {
      cache.set(workflow, null);
      return undefined;
    }
  };
};

export const createDeferredCallSavedWorkflowReconciler = (reconcile: () => void) => {
  let timer: ReturnType<typeof setTimeout> | null = null;

  return {
    dispose: () => {
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
    },
    schedule: () => {
      if (timer !== null) {
        return;
      }

      timer = setTimeout(() => {
        timer = null;
        reconcile();
      }, 0);
    },
  };
};

export const pruneStaleCallSavedWorkflowNodeState = <T,>(
  state: Map<string, T>,
  currentNodeIds: ReadonlySet<string>
): void => {
  for (const nodeId of state.keys()) {
    if (!currentNodeIds.has(nodeId)) {
      state.delete(nodeId);
    }
  }
};

type RetryableDetailWorkflowNode = { token: number; workflowId: string };

const hasSameFieldType = (left: unknown, right: unknown): boolean => {
  if (!left || !right || typeof left !== 'object' || typeof right !== 'object') {
    return false;
  }

  const leftType = (left as { type?: unknown }).type;
  const rightType = (right as { type?: unknown }).type;

  if (!leftType || !rightType || typeof leftType !== 'object' || typeof rightType !== 'object') {
    return false;
  }

  return (
    (leftType as { name?: unknown }).name === (rightType as { name?: unknown }).name &&
    (leftType as { cardinality?: unknown }).cardinality === (rightType as { cardinality?: unknown }).cardinality &&
    (leftType as { batch?: unknown }).batch === (rightType as { batch?: unknown }).batch
  );
};

const needsDynamicFieldSync = (
  node: Parameters<typeof getSavedWorkflowDynamicEdgeIdsToRemove>[0]['nodes'][number],
  fields: ReturnType<typeof getSavedWorkflowDynamicFields>,
  edgeIdsToRemove: string[],
  edges: Parameters<typeof getSavedWorkflowDynamicEdgeIdsToRemove>[0]['edges']
): boolean => {
  if (node.type !== 'invocation') {
    return false;
  }

  const currentTemplates = node.data.dynamicInputTemplates ?? {};
  const currentDynamicNames = new Set([
    ...Object.keys(currentTemplates),
    ...Object.keys(node.data.inputs).filter((name) => name.startsWith(CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX)),
  ]);

  if (currentDynamicNames.size !== fields.length || fields.some((field) => !currentDynamicNames.has(field.fieldName))) {
    return true;
  }

  if (
    fields.some((field) => {
      const currentTemplate = currentTemplates[field.fieldName];
      const currentInstance = node.data.inputs[field.fieldName];

      return (
        !currentTemplate ||
        !currentInstance ||
        !hasSameFieldType(currentTemplate, field.fieldTemplate) ||
        JSON.stringify(currentTemplate) !== JSON.stringify(field.fieldTemplate) ||
        (currentInstance.label === currentTemplate.title && currentInstance.label !== field.label) ||
        ((currentInstance.description ?? '') === currentTemplate.description &&
          (currentInstance.description ?? '') !== field.description)
      );
    })
  ) {
    return true;
  }

  return edgeIdsToRemove.some((edgeId) => edges.some((edge) => edge.id === edgeId));
};

/** Reconciles asynchronously loaded child workflow forms into the project document. */
export const CallSavedWorkflowSyncRuntime = () => {
  const queryClient = useQueryClient();
  const { commands, project: projectPort } = useWorkflowUi();
  const retryableDetailWorkflowIds = useRef(new Map<string, RetryableDetailWorkflowNode>());
  const retryableDetailWorkflowQueries = useRef(new Map<string, number>());
  const nextRetryToken = useRef(0);
  const previousDetailWorkflowIds = useRef(new Map<string, string>());
  const previousDetailStatuses = useRef(new Map<string, 'loading' | 'ready' | 'error'>());
  const parseChildWorkflow = useMemo(() => createSavedWorkflowDocumentParser(), []);
  const authorizeNodeRetry = (nodeId: string, workflowId: string) => {
    retryableDetailWorkflowIds.current.set(nodeId, { token: ++nextRetryToken.current, workflowId });
  };
  const authorizeWorkflowRetry = (workflowId: string) => {
    retryableDetailWorkflowQueries.current.set(workflowId, ++nextRetryToken.current);
  };
  const revokeNodeRetries = (workflowId?: string) => {
    for (const [nodeId, authorization] of retryableDetailWorkflowIds.current) {
      if (workflowId === undefined || authorization.workflowId === workflowId) {
        retryableDetailWorkflowIds.current.delete(nodeId);
      }
    }
  };
  const reconcile = () => {
    const templatesSnapshot = getInvocationTemplatesSnapshot();

    if (templatesSnapshot.status !== 'loaded') {
      return;
    }

    const document = projectPort.getSnapshot().projectGraph;
    const currentNodeIds = new Set(document.nodes.map((node) => node.id));
    const scheduledDetailFetches = new Set<string>();

    pruneStaleCallSavedWorkflowNodeState(retryableDetailWorkflowIds.current, currentNodeIds);
    pruneStaleCallSavedWorkflowNodeState(previousDetailStatuses.current, currentNodeIds);
    pruneStaleCallSavedWorkflowNodeState(previousDetailWorkflowIds.current, currentNodeIds);

    const setStatus = (nodeId: string, workflowId: string, status: 'loading' | 'ready' | 'error') => {
      const currentDocument = projectPort.getSnapshot().projectGraph;
      const currentNode = currentDocument.nodes.find((candidate) => candidate.id === nodeId);

      if (
        currentNode?.type === 'invocation' &&
        currentNode.data.type === 'call_saved_workflow' &&
        currentNode.data.inputs.workflow_id?.value === workflowId &&
        currentNode.data.callSavedWorkflowStatus !== status
      ) {
        commands.editGraph({ nodeId, status, type: 'setCallSavedWorkflowStatus' });
      }
    };

    for (const node of document.nodes) {
      if (node.type !== 'invocation' || node.data.type !== 'call_saved_workflow') {
        continue;
      }

      const workflowId =
        typeof node.data.inputs.workflow_id?.value === 'string' ? node.data.inputs.workflow_id.value : '';
      const previousWorkflowId = previousDetailWorkflowIds.current.get(node.id);
      const previousStatus = previousDetailStatuses.current.get(node.id);
      const currentStatus = node.data.callSavedWorkflowStatus ?? (workflowId ? 'loading' : 'ready');
      const detailOptions = workflowId ? savedWorkflowDetailQueryOptions(workflowId) : undefined;
      const query = detailOptions ? queryClient.getQueryCache().find({ queryKey: detailOptions.queryKey }) : undefined;

      previousDetailStatuses.current.set(node.id, currentStatus);

      previousDetailWorkflowIds.current.set(node.id, workflowId);

      if (!workflowId) {
        const hasDynamicFields =
          Object.keys(node.data.dynamicInputTemplates ?? {}).length > 0 ||
          Object.keys(node.data.inputs).some((name) => name.startsWith(CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX));

        if (hasDynamicFields || node.data.callSavedWorkflowStatus !== 'ready') {
          commands.editGraph({
            edgeIdsToRemove: [],
            fields: [],
            nodeId: node.id,
            status: 'ready',
            type: 'syncCallSavedWorkflowFields',
          });
        }
        continue;
      }

      if (!detailOptions) {
        continue;
      }

      if (
        (previousWorkflowId !== undefined && previousWorkflowId !== workflowId) ||
        (node.data.callSavedWorkflowStatus === 'loading' &&
          previousWorkflowId === workflowId &&
          previousStatus === 'error' &&
          query?.state.status === 'error')
      ) {
        authorizeNodeRetry(node.id, workflowId);
      }

      const nodeRetryAuthorization = retryableDetailWorkflowIds.current.get(node.id);
      const nodeRetryToken =
        nodeRetryAuthorization?.workflowId === workflowId ? nodeRetryAuthorization.token : undefined;
      const workflowRetryToken = retryableDetailWorkflowQueries.current.get(workflowId);
      const retryWasAuthorized = nodeRetryToken !== undefined || workflowRetryToken !== undefined;

      if (
        shouldFetchSavedWorkflowDetail(query, {
          retryErrors: retryWasAuthorized,
        })
      ) {
        if (scheduledDetailFetches.has(workflowId)) {
          continue;
        }
        scheduledDetailFetches.add(workflowId);

        const hasExistingDetail = query?.state.data !== undefined && query.state.data !== null;

        if (!hasExistingDetail) {
          setStatus(node.id, workflowId, 'loading');
        }

        void queryClient
          .fetchQuery(detailOptions)
          .then(() => {
            const currentNodeRetry = retryableDetailWorkflowIds.current.get(node.id);
            const currentWorkflowRetryToken = retryableDetailWorkflowQueries.current.get(workflowId);
            const hasNewerWorkflowRetry = currentWorkflowRetryToken !== workflowRetryToken;

            if (nodeRetryToken !== undefined && currentNodeRetry?.token === nodeRetryToken) {
              retryableDetailWorkflowIds.current.delete(node.id);
            }
            if (workflowRetryToken !== undefined && currentWorkflowRetryToken === workflowRetryToken) {
              retryableDetailWorkflowQueries.current.delete(workflowId);
            }

            if (hasNewerWorkflowRetry) {
              void queryClient.invalidateQueries({
                exact: true,
                queryKey: detailOptions.queryKey,
                refetchType: 'none',
              });
            }
          })
          .catch(() => {
            const currentNodeRetry = retryableDetailWorkflowIds.current.get(node.id);
            const currentWorkflowRetryToken = retryableDetailWorkflowQueries.current.get(workflowId);
            if (nodeRetryToken !== undefined && currentNodeRetry?.token === nodeRetryToken) {
              retryableDetailWorkflowIds.current.delete(node.id);
            }
            if (workflowRetryToken !== undefined && currentWorkflowRetryToken === workflowRetryToken) {
              retryableDetailWorkflowQueries.current.delete(workflowId);
            }
            const failedQuery = queryClient.getQueryCache().find({ queryKey: detailOptions.queryKey });
            const currentNode = projectPort
              .getSnapshot()
              .projectGraph.nodes.find((candidate) => candidate.id === node.id);
            const isCurrentWorkflow =
              currentNode?.type === 'invocation' &&
              currentNode.data.type === 'call_saved_workflow' &&
              currentNode.data.inputs.workflow_id?.value === workflowId;
            if (
              isCurrentWorkflow &&
              shouldRetrySavedWorkflowDetailAfterFailure(
                retryWasAuthorized,
                failedQuery?.state.data !== undefined && failedQuery.state.data !== null
              )
            ) {
              // Keep the node blocked while stale data is being retried. This
              // permits transient revalidation failures to recover without
              // allowing a deleted child workflow to enqueue stale inputs.
              authorizeNodeRetry(node.id, workflowId);
            }
            setStatus(node.id, workflowId, 'error');
          });
        continue;
      }

      const queryStatus = getSavedWorkflowDetailQueryStatus(query);

      if (queryStatus === 'loading') {
        setStatus(node.id, workflowId, 'loading');
        continue;
      }

      if (queryStatus === 'error') {
        setStatus(node.id, workflowId, 'error');
        continue;
      }

      const record = query?.state.data as WorkflowRecordDTO | undefined;
      const selectedWorkflow = record ? getSelectedSavedWorkflow(workflowId, record) : undefined;

      if (!selectedWorkflow || selectedWorkflow.call_saved_workflow_compatibility?.is_callable === false) {
        setStatus(node.id, workflowId, 'error');
        continue;
      }

      const childDocument = parseChildWorkflow(selectedWorkflow.workflow);

      if (!childDocument) {
        setStatus(node.id, workflowId, 'error');
        continue;
      }

      const fields = getSavedWorkflowDynamicFields(childDocument, templatesSnapshot.templates);
      const edgeIdsToRemove = getSavedWorkflowDynamicEdgeIdsToRemove(
        document,
        node.id,
        fields,
        templatesSnapshot.templates
      );

      if (
        node.data.callSavedWorkflowStatus !== 'ready' ||
        needsDynamicFieldSync(node, fields, edgeIdsToRemove, document.edges)
      ) {
        commands.editGraph({
          edgeIdsToRemove,
          fields,
          nodeId: node.id,
          status: 'ready',
          type: 'syncCallSavedWorkflowFields',
        });
      }
    }
  };

  /* eslint-disable react-hooks/rules-of-hooks -- useMountEffect is the repository's explicit useEffect wrapper */
  useMountEffect(() => {
    const reconciler = createDeferredCallSavedWorkflowReconciler(reconcile);
    reconciler.schedule();

    const unsubscribeProject = projectPort.subscribe(reconciler.schedule);
    const unsubscribeTemplates = subscribeInvocationTemplates(reconciler.schedule);
    const unsubscribeQueries = queryClient.getQueryCache().subscribe((event) => {
      if (isSavedWorkflowDetailQueryKey(event.query.queryKey)) {
        reconciler.schedule();
      }
    });
    const unsubscribeLibrary = onWorkflowLibraryCacheInvalidated((workflowId) => {
      if (workflowId) {
        revokeNodeRetries(workflowId);
        authorizeWorkflowRetry(workflowId);
        void queryClient.invalidateQueries({
          exact: true,
          queryKey: savedWorkflowDetailQueryKey(workflowId),
          refetchType: 'none',
        });
      } else {
        revokeNodeRetries();
        for (const query of queryClient.getQueryCache().findAll({ queryKey: ['workflow', 'call-saved', 'detail'] })) {
          if (isSavedWorkflowDetailQueryKey(query.queryKey)) {
            authorizeWorkflowRetry(query.queryKey[3]);
          }
        }

        void queryClient.invalidateQueries({
          queryKey: ['workflow', 'call-saved', 'detail'],
          refetchType: 'none',
        });
      }

      reconciler.schedule();
    });

    return () => {
      reconciler.dispose();
      unsubscribeProject();
      unsubscribeTemplates();
      unsubscribeQueries();
      unsubscribeLibrary();
    };
  });
  /* eslint-enable react-hooks/rules-of-hooks */

  return null;
};
