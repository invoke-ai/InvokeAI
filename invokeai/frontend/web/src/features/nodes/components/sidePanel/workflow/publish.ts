import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { $templates } from 'features/nodes/store/nodesSlice';
import {
  selectNodes,
  selectNodesSlice,
  selectWorkflowFormNodeFieldFieldIdentifiersDeduped,
  selectWorkflowId,
} from 'features/nodes/store/selectors';
import type { Templates } from 'features/nodes/store/types';
import type { FieldIdentifier } from 'features/nodes/types/field';
import { isBoardFieldType } from 'features/nodes/types/field';
import { isBatchNode, isGeneratorNode, isInvocationNode } from 'features/nodes/types/invocation';
import { atom, computed } from 'nanostores';
import { useMemo } from 'react';
import { useGetBatchStatusQuery } from 'services/api/endpoints/queue';
import { useGetWorkflowQuery } from 'services/api/endpoints/workflows';
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
export const $validationRunData = atom<{ batchId: string; workflowId: string } | null>(null);

export const useIsValidationRunInProgress = () => {
  const validationRunData = useStore($validationRunData);
  const { isValidationRunInProgress } = useGetBatchStatusQuery(
    validationRunData?.batchId ? { batch_id: validationRunData.batchId } : skipToken,
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
  return validationRunData !== null || isValidationRunInProgress;
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

const queryOptions = {
  selectFromResult: ({ currentData }) => {
    if (!currentData) {
      return { isPublished: false };
    }
    return { isPublished: currentData.is_published };
  },
} satisfies Parameters<typeof useGetWorkflowQuery>[1];

export const useIsWorkflowPublished = () => {
  const workflowId = useAppSelector(selectWorkflowId);
  const { isPublished } = useGetWorkflowQuery(workflowId ?? skipToken, queryOptions);

  return isPublished;
};

// These nodes are not allowed to be in published workflows because they dynamically generate model identifiers
const NODE_TYPE_PUBLISH_DENYLIST = [
  'metadata_to_model',
  'metadata_to_sdxl_model',
  'metadata_to_vae',
  'metadata_to_lora_collection',
  'metadata_to_loras',
  'metadata_to_sdlx_loras',
  'metadata_to_controlnets',
  'metadata_to_ip_adapters',
  'metadata_to_t2i_adapters',
];

export const selectHasUnpublishableNodes = createSelector(selectNodes, (nodes) => {
  for (const node of nodes) {
    if (!isInvocationNode(node)) {
      return true;
    }
    if (isBatchNode(node) || isGeneratorNode(node)) {
      return true;
    }
    if (NODE_TYPE_PUBLISH_DENYLIST.includes(node.data.type)) {
      return true;
    }
  }
  return false;
});
