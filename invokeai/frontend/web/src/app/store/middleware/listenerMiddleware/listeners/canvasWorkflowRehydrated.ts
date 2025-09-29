import type { AppStartListening } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { selectCanvasWorkflow } from 'features/controlLayers/store/canvasWorkflowSlice';
import { getFormFieldInitialValues } from 'features/nodes/store/nodesSlice';
import { SHARED_NODE_PROPERTIES } from 'features/nodes/types/constants';
import type { AnyNode } from 'features/nodes/types/invocation';
import { REMEMBER_REHYDRATED } from 'redux-remember';

/**
 * When the app rehydrates from storage, we need to populate the canvasWorkflowNodes
 * shadow slice if a canvas workflow was previously selected.
 *
 * This ensures that exposed fields are visible when the page loads with a workflow already selected.
 */
export const addCanvasWorkflowRehydratedListener = (startListening: AppStartListening) => {
  startListening({
    type: REMEMBER_REHYDRATED,
    effect: async (_action, { dispatch, getState }) => {
      const state = getState();
      const { workflow, inputNodeId } = state.canvasWorkflow;

      // If there's a canvas workflow already selected, we need to load it into shadow nodes
      if (workflow && inputNodeId) {
        // Manually dispatch the fulfilled action to populate shadow nodes
        // We can't use the thunk because the workflow is already loaded
        dispatch({
          type: selectCanvasWorkflow.fulfilled.type,
          payload: {
            workflow,
            inputNodeId,
            outputNodeId: state.canvasWorkflow.outputNodeId,
            workflowId: state.canvasWorkflow.selectedWorkflowId,
            fieldValues: state.canvasWorkflow.fieldValues,
          },
        });
      }
    },
  });
};