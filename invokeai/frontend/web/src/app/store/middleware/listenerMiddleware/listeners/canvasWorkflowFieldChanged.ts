import type { ActionCreatorWithPayload } from '@reduxjs/toolkit';
import type { AppStartListening, RootState } from 'app/store/store';
import * as canvasWorkflowNodesActions from 'features/controlLayers/store/canvasWorkflowNodesSlice';
import * as nodesActions from 'features/nodes/store/nodesSlice';
import type { AnyNode } from 'features/nodes/types/invocation';

/**
 * Listens for field value changes on nodes and redirects them to the canvas workflow nodes slice
 * if the node belongs to a canvas workflow.
 */
export const addCanvasWorkflowFieldChangedListener = (startListening: AppStartListening) => {
  // List of all field mutation actions from nodesSlice with their canvas workflow counterparts
  const fieldMutationActionPairs: Array<{
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    nodesAction: ActionCreatorWithPayload<any>;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    canvasAction: ActionCreatorWithPayload<any>;
  }> = [
    {
      nodesAction: nodesActions.fieldStringValueChanged,
      canvasAction: canvasWorkflowNodesActions.fieldStringValueChanged,
    },
    {
      nodesAction: nodesActions.fieldIntegerValueChanged,
      canvasAction: canvasWorkflowNodesActions.fieldIntegerValueChanged,
    },
    {
      nodesAction: nodesActions.fieldFloatValueChanged,
      canvasAction: canvasWorkflowNodesActions.fieldFloatValueChanged,
    },
    {
      nodesAction: nodesActions.fieldBooleanValueChanged,
      canvasAction: canvasWorkflowNodesActions.fieldBooleanValueChanged,
    },
    {
      nodesAction: nodesActions.fieldModelIdentifierValueChanged,
      canvasAction: canvasWorkflowNodesActions.fieldModelIdentifierValueChanged,
    },
    {
      nodesAction: nodesActions.fieldEnumModelValueChanged,
      canvasAction: canvasWorkflowNodesActions.fieldEnumModelValueChanged,
    },
    {
      nodesAction: nodesActions.fieldSchedulerValueChanged,
      canvasAction: canvasWorkflowNodesActions.fieldSchedulerValueChanged,
    },
    {
      nodesAction: nodesActions.fieldBoardValueChanged,
      canvasAction: canvasWorkflowNodesActions.fieldBoardValueChanged,
    },
    {
      nodesAction: nodesActions.fieldImageValueChanged,
      canvasAction: canvasWorkflowNodesActions.fieldImageValueChanged,
    },
    {
      nodesAction: nodesActions.fieldColorValueChanged,
      canvasAction: canvasWorkflowNodesActions.fieldColorValueChanged,
    },
    {
      nodesAction: nodesActions.fieldImageCollectionValueChanged,
      canvasAction: canvasWorkflowNodesActions.fieldImageCollectionValueChanged,
    },
    {
      nodesAction: nodesActions.fieldStringCollectionValueChanged,
      canvasAction: canvasWorkflowNodesActions.fieldStringCollectionValueChanged,
    },
    {
      nodesAction: nodesActions.fieldIntegerCollectionValueChanged,
      canvasAction: canvasWorkflowNodesActions.fieldIntegerCollectionValueChanged,
    },
    {
      nodesAction: nodesActions.fieldFloatCollectionValueChanged,
      canvasAction: canvasWorkflowNodesActions.fieldFloatCollectionValueChanged,
    },
    {
      nodesAction: nodesActions.fieldFloatGeneratorValueChanged,
      canvasAction: canvasWorkflowNodesActions.fieldFloatGeneratorValueChanged,
    },
    {
      nodesAction: nodesActions.fieldIntegerGeneratorValueChanged,
      canvasAction: canvasWorkflowNodesActions.fieldIntegerGeneratorValueChanged,
    },
    {
      nodesAction: nodesActions.fieldStringGeneratorValueChanged,
      canvasAction: canvasWorkflowNodesActions.fieldStringGeneratorValueChanged,
    },
    {
      nodesAction: nodesActions.fieldImageGeneratorValueChanged,
      canvasAction: canvasWorkflowNodesActions.fieldImageGeneratorValueChanged,
    },
    { nodesAction: nodesActions.fieldValueReset, canvasAction: canvasWorkflowNodesActions.fieldValueReset },
  ];

  for (const { nodesAction, canvasAction } of fieldMutationActionPairs) {
    startListening({
      actionCreator: nodesAction,
      effect: (action, { dispatch, getState }) => {
        const state = getState() as RootState;
        const { nodeId } = action.payload;

        // Check if this node exists in canvas workflow nodes
        const canvasWorkflowNode = state.canvasWorkflowNodes.nodes.find((n: AnyNode) => n.id === nodeId);
        const regularNode = state.nodes.present.nodes.find((n: AnyNode) => n.id === nodeId);

        console.log('[canvasWorkflowFieldChanged] Field changed:', {
          nodeId,
          hasCanvasNode: !!canvasWorkflowNode,
          hasRegularNode: !!regularNode,
          action: action.type,
          payload: action.payload,
        });

        // If the node exists in canvas workflow, redirect the action
        // This ensures canvas workflow fields always update the canvas workflow nodes slice
        if (canvasWorkflowNode) {
          console.log('[canvasWorkflowFieldChanged] Redirecting to canvas workflow nodes:', canvasAction.type);
          dispatch(canvasAction(action.payload));
        }
      },
    });
  }
};
