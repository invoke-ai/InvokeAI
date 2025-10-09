import { logger } from 'app/logging/logger';
import { useAppDispatch } from 'app/store/storeHooks';
import { createZodErrorSummary, logZodError } from 'common/util/zodErrorLogger';
import { getIsFormEmpty } from 'features/nodes/components/sidePanel/builder/form-manipulation';
import { $nodeExecutionStates } from 'features/nodes/hooks/useNodeExecutionState';
import { $templates, workflowLoaded } from 'features/nodes/store/nodesSlice';
import { $needsFit } from 'features/nodes/store/reactFlowInstance';
import { workflowModeChanged } from 'features/nodes/store/workflowLibrarySlice';
import { WorkflowMigrationError, WorkflowVersionError } from 'features/nodes/types/error';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { validateWorkflow } from 'features/nodes/util/workflow/validateWorkflow';
import { toast } from 'features/toast/toast';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { VIEWER_PANEL_ID, WORKSPACE_PANEL_ID } from 'features/ui/layouts/shared';
import { t } from 'i18next';
import { useCallback } from 'react';
import { serializeError } from 'serialize-error';
import { checkBoardAccess, checkImageAccess, checkModelAccess } from 'services/api/hooks/accessChecks';
import { z } from 'zod';

const log = logger('workflows');

/**
 * This hook manages the lower-level workflow validation and loading process.
 *
 * You probably should instead use `useLoadWorkflowWithDialog`, which opens a dialog to prevent loss of unsaved changes
 * and handles the loading process.
 *
 * Internally, `useLoadWorkflowWithDialog` uses these hooks...
 *
 * - `useLoadWorkflowFromFile`
 * - `useLoadWorkflowFromImage`
 * - `useLoadWorkflowFromLibrary`
 * - `useLoadWorkflowFromObject`
 *
 * ...each of which internally uses hook.
 */
export const useValidateAndLoadWorkflow = () => {
  const dispatch = useAppDispatch();
  const validateAndLoadWorkflow = useCallback(
    /**
     * Validate and load a workflow into the editor.
     *
     * The unvalidated workflow should be a JS object. Do not pass a raw JSON string.
     *
     * This function catches all errors. It toasts and logs on success and error.
     */
    async (
      unvalidatedWorkflow: unknown,
      origin: 'file' | 'image' | 'object' | 'library'
    ): Promise<WorkflowV3 | null> => {
      try {
        const templates = $templates.get();
        const { workflow, warnings } = await validateWorkflow({
          workflow: unvalidatedWorkflow,
          templates,
          checkImageAccess,
          checkBoardAccess,
          checkModelAccess,
        });

        if (origin !== 'library') {
          // Workflow IDs should always map directly to the workflow in the library. If the workflow is loaded from
          // some other source, and has an ID, we should remove it to ensure the app does not treat it as a library workflow.
          // For example, when saving a workflow, we might accidentally attempt to save instead of save-as.
          delete workflow.id;
        }

        $nodeExecutionStates.set({});
        dispatch(workflowLoaded(workflow));

        // If the form is empty, assume the user is editing a new workflow.
        if (getIsFormEmpty(workflow.form)) {
          dispatch(workflowModeChanged('edit'));
          navigationApi.focusPanel('workflows', WORKSPACE_PANEL_ID);
        } else {
          // Else assume they want to use the linear view of the workflow.
          dispatch(workflowModeChanged('view'));
          navigationApi.focusPanel('workflows', VIEWER_PANEL_ID);
        }

        if (!warnings.length) {
          toast({
            id: 'WORKFLOW_LOADED',
            title: t('toast.workflowLoaded'),
            status: 'success',
          });
        } else {
          toast({
            id: 'WORKFLOW_LOADED',
            title: t('toast.loadedWithWarnings'),
            status: 'warning',
          });

          warnings.forEach(({ message, ...rest }) => {
            log.warn(rest, message);
          });
        }

        $needsFit.set(true);
        return workflow;
      } catch (e) {
        if (e instanceof WorkflowVersionError) {
          // The workflow version was not recognized in the valid list of versions
          log.error({ error: serializeError(e) }, e.message);
          toast({
            id: 'UNABLE_TO_VALIDATE_WORKFLOW',
            title: t('nodes.unableToValidateWorkflow'),
            status: 'error',
            description: e.message,
          });
        } else if (e instanceof WorkflowMigrationError) {
          // There was a problem migrating the workflow to the latest version
          log.error({ error: serializeError(e) }, e.message);
          toast({
            id: 'UNABLE_TO_VALIDATE_WORKFLOW',
            title: t('nodes.unableToValidateWorkflow'),
            status: 'error',
            description: e.message,
          });
        } else if (e instanceof z.ZodError) {
          // There was a problem validating the workflow itself

          // Log comprehensive error information for developers
          logZodError(e, { context: 'Workflow validation' });

          // Create human-readable message for toast with node-specific formatting
          const pathFormatter = (path: string, message: string) => {
            const nodeMatch = path.match(/nodes\.(\d+)\.data\.inputs\.([^.]+)(?:\.(.+))?/);
            const nodeDataMatch = path.match(/nodes\.(\d+)\.data\.([^.]+)/);
            const nodeMatch2 = path.match(/nodes\.(\d+)\.([^.]+)/);

            if (nodeMatch) {
              const [, nodeIndex, fieldName, subField] = nodeMatch;
              const fieldPath = subField ? `${fieldName}.${subField}` : fieldName;
              return `Node ${parseInt(nodeIndex, 10) + 1} input "${fieldPath}": ${message}`;
            } else if (nodeDataMatch) {
              const [, nodeIndex, fieldName] = nodeDataMatch;
              return `Node ${parseInt(nodeIndex, 10) + 1} data "${fieldName}": ${message}`;
            } else if (nodeMatch2) {
              const [, nodeIndex, fieldName] = nodeMatch2;
              return `Node ${parseInt(nodeIndex, 10) + 1} "${fieldName}": ${message}`;
            } else if (path.startsWith('nodes.')) {
              const pathParts = path.split('.');
              const nodeIndex = pathParts[1];
              const remainingPath = pathParts.slice(2).join('.');
              return `Node ${parseInt(nodeIndex, 10) + 1} (${remainingPath || 'general'}): ${message}`;
            } else {
              return `${path || 'Workflow'}: ${message}`;
            }
          };

          const errorSummary = createZodErrorSummary(e, 10, pathFormatter);
          const message = `${t('nodes.workflowValidation')}: ${errorSummary}`;

          log.error(
            {
              error: serializeError(e),
              totalIssues: e.issues.length,
            },
            `Workflow validation failed with ${e.issues.length} issues: ${message}`
          );
          toast({
            id: 'UNABLE_TO_VALIDATE_WORKFLOW',
            title: t('nodes.unableToValidateWorkflow'),
            status: 'error',
            description: message,
          });
        } else {
          // Some other error occurred
          log.error({ error: serializeError(e as Error) }, t('nodes.unknownErrorValidatingWorkflow'));
          toast({
            id: 'UNABLE_TO_VALIDATE_WORKFLOW',
            title: t('nodes.unableToValidateWorkflow'),
            status: 'error',
            description: t('nodes.unknownErrorValidatingWorkflow'),
          });
        }
        return null;
      }
    },
    [dispatch]
  );

  return validateAndLoadWorkflow;
};
