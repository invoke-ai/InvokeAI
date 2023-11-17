import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import { workflowLoadRequested } from 'features/nodes/store/actions';
import { workflowLoaded } from 'features/nodes/store/nodesSlice';
import { $flow } from 'features/nodes/store/reactFlowInstance';
import { WorkflowVersionError } from 'features/nodes/types/error';
import { validateWorkflow } from 'features/nodes/util/validateWorkflow';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { t } from 'i18next';
import { z } from 'zod';
import { fromZodError } from 'zod-validation-error';
import { startAppListening } from '..';

export const addWorkflowLoadRequestedListener = () => {
  startAppListening({
    actionCreator: workflowLoadRequested,
    effect: (action, { dispatch, getState }) => {
      const log = logger('nodes');
      const workflow = action.payload;
      const nodeTemplates = getState().nodes.nodeTemplates;

      try {
        const { workflow: validatedWorkflow, warnings } = validateWorkflow(
          workflow,
          nodeTemplates
        );
        dispatch(workflowLoaded(validatedWorkflow));
        if (!warnings.length) {
          dispatch(
            addToast(
              makeToast({
                title: t('toast.workflowLoaded'),
                status: 'success',
              })
            )
          );
        } else {
          dispatch(
            addToast(
              makeToast({
                title: t('toast.loadedWithWarnings'),
                status: 'warning',
              })
            )
          );
          warnings.forEach(({ message, ...rest }) => {
            log.warn(rest, message);
          });
        }

        dispatch(setActiveTab('nodes'));
        requestAnimationFrame(() => {
          $flow.get()?.fitView();
        });
      } catch (e) {
        if (e instanceof WorkflowVersionError) {
          // The workflow version was not recognized in the valid list of versions
          log.error({ error: parseify(e) }, e.message);
          dispatch(
            addToast(
              makeToast({
                title: t('nodes.unableToValidateWorkflow'),
                status: 'error',
                description: e.message,
              })
            )
          );
        } else if (e instanceof z.ZodError) {
          // There was a problem validating the workflow itself
          const { message } = fromZodError(e, {
            prefix: t('nodes.workflowValidation'),
          });
          log.error({ error: parseify(e) }, message);
          dispatch(
            addToast(
              makeToast({
                title: t('nodes.unableToValidateWorkflow'),
                status: 'error',
                description: message,
              })
            )
          );
        } else {
          // Some other error occurred
          console.log(e);
          log.error(
            { error: parseify(e) },
            t('nodes.unknownErrorValidatingWorkflow')
          );
          dispatch(
            addToast(
              makeToast({
                title: t('nodes.unableToValidateWorkflow'),
                status: 'error',
                description: t('nodes.unknownErrorValidatingWorkflow'),
              })
            )
          );
        }
      }
    },
  });
};
