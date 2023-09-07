import { logger } from 'app/logging/logger';
import { workflowLoadRequested } from 'features/nodes/store/actions';
import { workflowLoaded } from 'features/nodes/store/nodesSlice';
import { $flow } from 'features/nodes/store/reactFlowInstance';
import { validateWorkflow } from 'features/nodes/util/validateWorkflow';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { startAppListening } from '..';

export const addWorkflowLoadedListener = () => {
  startAppListening({
    actionCreator: workflowLoadRequested,
    effect: (action, { dispatch, getState }) => {
      const log = logger('nodes');
      const workflow = action.payload;
      const nodeTemplates = getState().nodes.nodeTemplates;

      const { workflow: validatedWorkflow, errors } = validateWorkflow(
        workflow,
        nodeTemplates
      );

      dispatch(workflowLoaded(validatedWorkflow));

      if (!errors.length) {
        dispatch(
          addToast(
            makeToast({
              title: 'Workflow Loaded',
              status: 'success',
            })
          )
        );
      } else {
        dispatch(
          addToast(
            makeToast({
              title: 'Workflow Loaded with Warnings',
              status: 'warning',
            })
          )
        );
        errors.forEach(({ message, ...rest }) => {
          log.warn(rest, message);
        });
      }

      dispatch(setActiveTab('nodes'));
      requestAnimationFrame(() => {
        $flow.get()?.fitView();
      });
    },
  });
};
