import type { AppStartListening } from 'app/store/store';
import * as canvasWorkflowNodesActions from 'features/controlLayers/store/canvasWorkflowNodesSlice';
import * as nodesActions from 'features/nodes/store/nodesSlice';

/**
 * Listens for field value changes on nodes and redirects them to the canvas workflow nodes slice
 * if the node belongs to a canvas workflow (exists in canvasWorkflowNodes but not in nodes).
 */
export const addCanvasWorkflowFieldChangedListener = (startListening: AppStartListening) => {
  // List of all field mutation actions from nodesSlice
  const fieldMutationActions = [
    nodesActions.fieldStringValueChanged,
    nodesActions.fieldIntegerValueChanged,
    nodesActions.fieldFloatValueChanged,
    nodesActions.fieldBooleanValueChanged,
    nodesActions.fieldModelIdentifierValueChanged,
    nodesActions.fieldEnumModelValueChanged,
    nodesActions.fieldSchedulerValueChanged,
    nodesActions.fieldBoardValueChanged,
    nodesActions.fieldImageValueChanged,
    nodesActions.fieldColorValueChanged,
    nodesActions.fieldImageCollectionValueChanged,
    nodesActions.fieldStringCollectionValueChanged,
    nodesActions.fieldIntegerCollectionValueChanged,
    nodesActions.fieldFloatCollectionValueChanged,
    nodesActions.fieldFloatGeneratorValueChanged,
    nodesActions.fieldIntegerGeneratorValueChanged,
    nodesActions.fieldStringGeneratorValueChanged,
    nodesActions.fieldImageGeneratorValueChanged,
    nodesActions.fieldValueReset,
  ];

  for (const actionCreator of fieldMutationActions) {
    startListening({
      actionCreator,
      effect: (action: any, { dispatch, getState }: any) => {
        const state = getState();
        const { nodeId } = action.payload;

        // Check if this node exists in canvas workflow nodes
        const canvasWorkflowNode = state.canvasWorkflowNodes.nodes.find((n: any) => n.id === nodeId);
        const regularNode = state.nodes.present.nodes.find((n: any) => n.id === nodeId);

        // If the node exists in canvas workflow but NOT in regular nodes, redirect the action
        if (canvasWorkflowNode && !regularNode) {
          // Get the corresponding action from canvasWorkflowNodesSlice
          const actionName = actionCreator.type.split('/').pop() as keyof typeof canvasWorkflowNodesActions;
          const canvasWorkflowAction = canvasWorkflowNodesActions[actionName];

          if (canvasWorkflowAction && typeof canvasWorkflowAction === 'function') {
            dispatch(canvasWorkflowAction(action.payload as any));
          }
        }
      },
    });
  }
};