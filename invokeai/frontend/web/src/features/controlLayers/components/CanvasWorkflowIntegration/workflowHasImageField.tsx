import { logger } from 'app/logging/logger';
import { $templates } from 'features/nodes/store/nodesSlice';
import { isNodeFieldElement } from 'features/nodes/types/workflow';
import type { paths } from 'services/api/schema';

const log = logger('canvas-workflow-integration');

type WorkflowResponse =
  paths['/api/v1/workflows/i/{workflow_id}']['get']['responses']['200']['content']['application/json'];

/**
 * Checks if a workflow is compatible with canvas workflow integration.
 * Requirements:
 * 1. Has a Form Builder (allows users to modify parameters)
 * 2. Has a canvas_output node (explicit canvas output target)
 * 3. Has at least one ImageField in the Form Builder (receives the canvas image)
 * @param workflow The workflow to check
 * @returns true if the workflow meets all requirements, false otherwise
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

  // Must have a canvas_output node to define where the output image goes
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const hasCanvasOutputNode = workflow.workflow.nodes?.some((n: any) => (n.data as any)?.type === 'canvas_output');
  if (!hasCanvasOutputNode) {
    log.debug('Workflow has no canvas_output node - excluding from list');
    return false;
  }

  const templates = $templates.get();
  const elements = workflow.workflow.form.elements;

  log.debug('Workflow has form builder and canvas_output node, checking form elements for ImageField');

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
