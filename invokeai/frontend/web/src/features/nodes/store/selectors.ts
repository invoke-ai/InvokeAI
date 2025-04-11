import type { Selector } from '@reduxjs/toolkit';
import { createSelector } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import { getElement } from 'features/nodes/components/sidePanel/builder/form-manipulation';
import type { NodesState } from 'features/nodes/store/types';
import type { FieldInputInstance } from 'features/nodes/types/field';
import type { AnyNode, InvocationNode, InvocationNodeData } from 'features/nodes/types/invocation';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { isContainerElement, isNodeFieldElement } from 'features/nodes/types/workflow';
import { uniqBy } from 'lodash-es';
import { assert } from 'tsafe';

export const selectNode = (nodesSlice: NodesState, nodeId: string): AnyNode => {
  const node = nodesSlice.nodes.find((node) => node.id === nodeId);
  assert(node !== undefined, `Node ${nodeId} not found`);
  return node;
};

export const selectInvocationNode = (nodesSlice: NodesState, nodeId: string): InvocationNode => {
  const node = nodesSlice.nodes.find((node) => node.id === nodeId);
  assert(isInvocationNode(node), `Node ${nodeId} is not an invocation node`);
  return node;
};

export const selectInvocationNodeSafe = (nodesSlice: NodesState, nodeId: string): InvocationNode | undefined => {
  const node = nodesSlice.nodes.find((node) => node.id === nodeId);
  if (!isInvocationNode(node)) {
    return undefined;
  }
  return node;
};

export const selectNodeData = (nodesSlice: NodesState, nodeId: string): InvocationNodeData => {
  const node = selectInvocationNode(nodesSlice, nodeId);
  return node.data;
};

export const selectFieldInputInstance = (
  nodesSlice: NodesState,
  nodeId: string,
  fieldName: string
): FieldInputInstance => {
  const data = selectNodeData(nodesSlice, nodeId);
  const field = data.inputs[fieldName];
  assert(field !== undefined, `Field ${fieldName} not found in node ${nodeId}`);
  return field;
};

export const selectFieldInputInstanceSafe = (
  nodesSlice: NodesState,
  nodeId: string,
  fieldName: string
): FieldInputInstance | null => {
  const data = selectNodeData(nodesSlice, nodeId);
  return data.inputs[fieldName] ?? null;
};

export const selectLastSelectedNode = (nodesSlice: NodesState) => {
  const selectedNodes = nodesSlice.nodes.filter((n) => n.selected);
  if (selectedNodes.length === 1) {
    return selectedNodes[0];
  }
  return null;
};

export const selectNodesSlice = (state: RootState) => state.nodes.present;

export const selectLastSelectedNodeId = createSelector(selectNodesSlice, ({ nodes }) => {
  const selectedNodes = nodes.filter(isInvocationNode).filter((n) => n.selected);
  if (selectedNodes.length === 1) {
    return selectedNodes[0]?.id;
  }
  return null;
});

const createNodesSelector = <T>(selector: Selector<NodesState, T>) => createSelector(selectNodesSlice, selector);
export const selectNodes = createNodesSelector((nodes) => nodes.nodes);
export const selectEdges = createNodesSelector((nodes) => nodes.edges);
export const selectMayUndo = createSelector(
  (state: RootState) => state.nodes,
  (nodes) => nodes.past.length > 0
);
export const selectMayRedo = createSelector(
  (state: RootState) => state.nodes,
  (nodes) => nodes.future.length > 0
);

export const selectWorkflowName = createNodesSelector((nodes) => nodes.name);

export const selectWorkflowId = createNodesSelector((workflow) => workflow.id);
export const selectWorkflowDescription = createNodesSelector((workflow) => workflow.description);
export const selectWorkflowNotes = createNodesSelector((workflow) => workflow.notes);
export const selectWorkflowAuthor = createNodesSelector((workflow) => workflow.author);
export const selectWorkflowContact = createNodesSelector((workflow) => workflow.contact);
export const selectWorkflowTags = createNodesSelector((workflow) => workflow.tags);
export const selectWorkflowVersion = createNodesSelector((workflow) => workflow.version);
export const selectWorkflowForm = createNodesSelector((workflow) => workflow.form);

export const selectFormRootElementId = createNodesSelector((workflow) => {
  return workflow.form.rootElementId;
});
export const selectFormRootElement = createNodesSelector((workflow) => {
  return getElement(workflow.form, workflow.form.rootElementId, isContainerElement);
});
export const selectIsFormEmpty = createNodesSelector((workflow) => {
  const rootElement = workflow.form.elements[workflow.form.rootElementId];
  if (!rootElement || !isContainerElement(rootElement)) {
    return true;
  }
  return rootElement.data.children.length === 0;
});
export const selectFormInitialValues = createNodesSelector((workflow) => workflow.formFieldInitialValues);
export const selectNodeFieldElements = createNodesSelector((workflow) =>
  Object.values(workflow.form.elements).filter(isNodeFieldElement)
);
export const selectWorkflowFormNodeFieldFieldIdentifiersDeduped = createSelector(
  selectNodeFieldElements,
  (nodeFieldElements) =>
    uniqBy(nodeFieldElements, (el) => `${el.data.fieldIdentifier.nodeId}-${el.data.fieldIdentifier.fieldName}`).map(
      (el) => el.data.fieldIdentifier
    )
);

export const buildSelectElement = (id: string) => createNodesSelector((workflow) => workflow.form?.elements[id]);
