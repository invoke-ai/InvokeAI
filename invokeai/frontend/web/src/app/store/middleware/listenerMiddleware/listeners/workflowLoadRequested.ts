import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { parseify } from 'common/util/serialize';
import { workflowLoaded, workflowLoadRequested } from 'features/nodes/store/actions';
import { $flow } from 'features/nodes/store/reactFlowInstance';
import { WorkflowMigrationError, WorkflowVersionError } from 'features/nodes/types/error';
import { validateWorkflow } from 'features/nodes/util/workflow/validateWorkflow';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { t } from 'i18next';
import { z } from 'zod';
import { fromZodError } from 'zod-validation-error';

export const addWorkflowLoadRequestedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: workflowLoadRequested,
    effect: (action, { dispatch, getState }) => {
      const log = logger('nodes');
      const { workflow, asCopy } = action.payload;
      const nodeTemplates = getState().nodes.templates;

      try {
        const { workflow: validatedWorkflow, warnings } = validateWorkflow(workflow, nodeTemplates);

        if (asCopy) {
          // If we're loading a copy, we need to remove the ID so that the backend will create a new workflow
          delete validatedWorkflow.id;
        }

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
        } else if (e instanceof WorkflowMigrationError) {
          // There was a problem migrating the workflow to the latest version
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
          log.error({ error: parseify(e) }, t('nodes.unknownErrorValidatingWorkflow'));
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
