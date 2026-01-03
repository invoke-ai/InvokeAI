import { logger } from 'app/logging/logger';
import { $templates } from 'features/nodes/store/nodesSlice';
import { isNodeFieldElement } from 'features/nodes/types/workflow';
import type { paths } from 'services/api/schema';

const log = logger('canvas-workflow-integration');

type WorkflowResponse =
  paths['/api/v1/workflows/i/{workflow_id}']['get']['responses']['200']['content']['application/json'];

/**
 * Checks if a workflow has at least one ImageField input exposed in the Form Builder.
 * Only workflows with Form Builder are supported, as they allow users to modify
 * models and other parameters. Workflows without Form Builder are excluded.
 * @param workflow The workflow to check
 * @returns true if the workflow has a Form Builder with at least one ImageField, false otherwise
 */
export function workflowHasImageField(workflow: WorkflowResponse | undefined): boolean {
  if (!workflow?.workflow) {
    log.debug('No workflow data provided');
    return false;
  }

  // Only workflows with Form Builder are supported
  // Workflows without Form Builder don't allow changing models and other parameters
  if (!workflow.workflow.form?.elements) {
    log.debug('Workflow has no form builder - excluding from list');
    return false;
  }

  const templates = $templates.get();
  const elements = workflow.workflow.form.elements;

  log.debug('Workflow has form builder, checking form elements for ImageField');

  for (const [elementId, element] of Object.entries(elements)) {
    if (isNodeFieldElement(element)) {
      const { fieldIdentifier } = element.data;

      // Find the node that contains this field
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const node = workflow.workflow.nodes?.find((n: any) => n.data?.id === fieldIdentifier.nodeId);
      if (!node) {
        continue;
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const nodeType = (node.data as any)?.type;
      if (!nodeType) {
        continue;
      }

      const template = templates[nodeType];
      if (!template?.inputs) {
        continue;
      }

      const fieldTemplate = template.inputs[fieldIdentifier.fieldName];
      if (!fieldTemplate) {
        continue;
      }

      // Check if this is an ImageField
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const fieldType = (fieldTemplate as any).type?.name;
      if (fieldType === 'ImageField') {
        log.debug({ elementId, fieldName: fieldIdentifier.fieldName }, 'Found ImageField in workflow form');
        return true;
      }
    }
  }

  // If we have a form but no ImageFields were found in it, return false
  log.debug('Workflow has form builder but no ImageField found in form elements');
  return false;
}
