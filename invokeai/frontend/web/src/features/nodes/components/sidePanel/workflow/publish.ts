import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import type { Templates } from 'features/nodes/store/types';
import { selectWorkflowFormNodeFieldFieldIdentifiersDeduped } from 'features/nodes/store/workflowSlice';
import type { FieldIdentifier } from 'features/nodes/types/field';
import { isBoardFieldType } from 'features/nodes/types/field';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { atom, computed } from 'nanostores';
import { useMemo } from 'react';
import { useGetBatchStatusQuery } from 'services/api/endpoints/queue';
import { assert } from 'tsafe';

export const $isInPublishFlow = atom(false);
export const $outputNodeId = atom<string | null>(null);
export const $isSelectingOutputNode = atom(false);
export const $isReadyToDoValidationRun = computed(
  [$isInPublishFlow, $outputNodeId, $isSelectingOutputNode],
  (isInPublishFlow, outputNodeId, isSelectingOutputNode) => {
    return isInPublishFlow && outputNodeId !== null && !isSelectingOutputNode;
  }
);
export const $validationRunBatchId = atom<string | null>(null);

export const useIsValidationRunInProgress = () => {
  const validationRunBatchId = useStore($validationRunBatchId);
  const { isValidationRunInProgress } = useGetBatchStatusQuery(
    validationRunBatchId ? { batch_id: validationRunBatchId } : skipToken,
    {
      selectFromResult: ({ currentData }) => {
        if (!currentData) {
          return { isValidationRunInProgress: false };
        }
        if (currentData && currentData.in_progress > 0) {
          return { isValidationRunInProgress: true };
        }
        return { isValidationRunInProgress: false };
      },
    }
  );
  return validationRunBatchId !== null || isValidationRunInProgress;
};

export const selectFieldIdentifiersWithInvocationTypes = createSelector(
  selectWorkflowFormNodeFieldFieldIdentifiersDeduped,
  selectNodesSlice,
  (fieldIdentifiers, nodes) => {
    const result: { nodeId: string; fieldName: string; type: string }[] = [];
    for (const fieldIdentifier of fieldIdentifiers) {
      const node = nodes.nodes.find((node) => node.id === fieldIdentifier.nodeId);
      assert(isInvocationNode(node), `Node ${fieldIdentifier.nodeId} not found`);
      result.push({ nodeId: fieldIdentifier.nodeId, fieldName: fieldIdentifier.fieldName, type: node.data.type });
    }

    return result;
  }
);

export const getPublishInputs = (fieldIdentifiers: (FieldIdentifier & { type: string })[], templates: Templates) => {
  // Certain field types are not allowed to be input fields on a published workflow
  const publishable: FieldIdentifier[] = [];
  const unpublishable: FieldIdentifier[] = [];
  for (const fieldIdentifier of fieldIdentifiers) {
    const fieldTemplate = templates[fieldIdentifier.type]?.inputs[fieldIdentifier.fieldName];
    if (!fieldTemplate) {
      unpublishable.push(fieldIdentifier);
      continue;
    }
    if (isBoardFieldType(fieldTemplate.type)) {
      unpublishable.push(fieldIdentifier);
      continue;
    }
    publishable.push(fieldIdentifier);
  }
  return { publishable, unpublishable };
};

export const usePublishInputs = () => {
  const templates = useStore($templates);
  const fieldIdentifiersWithInvocationTypes = useAppSelector(selectFieldIdentifiersWithInvocationTypes);
  const fieldIdentifiers = useMemo(
    () => getPublishInputs(fieldIdentifiersWithInvocationTypes, templates),
    [fieldIdentifiersWithInvocationTypes, templates]
  );

  return fieldIdentifiers;
};
