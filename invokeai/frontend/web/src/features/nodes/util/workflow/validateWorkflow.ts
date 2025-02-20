import { parseify } from 'common/util/serialize';
import { addElement, getIsFormEmpty } from 'features/nodes/components/sidePanel/builder/form-manipulation';
import type { Templates } from 'features/nodes/store/types';
import {
  isBoardFieldInputInstance,
  isImageFieldCollectionInputInstance,
  isImageFieldInputInstance,
  isModelFieldType,
  isModelIdentifierFieldInputInstance,
} from 'features/nodes/types/field';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import {
  buildNodeFieldElement,
  getDefaultForm,
  isNodeFieldElement,
  isWorkflowInvocationNode,
} from 'features/nodes/types/workflow';
import { getNeedsUpdate, updateNode } from 'features/nodes/util/node/nodeUpdate';
import { t } from 'i18next';
import type { JsonObject } from 'type-fest';

import { parseAndMigrateWorkflow } from './migrations';

type WorkflowWarning = {
  message: string;
  issues?: string[];
  data: JsonObject;
};

type ValidateWorkflowArgs = {
  workflow: unknown;
  templates: Templates;
  checkImageAccess: (name: string) => Promise<boolean>;
  checkBoardAccess: (id: string) => Promise<boolean>;
  checkModelAccess: (key: string) => Promise<boolean>;
};

type ValidateWorkflowResult = {
  workflow: WorkflowV3;
  warnings: WorkflowWarning[];
};

/**
 * Parses and validates a workflow:
 * - Parses the workflow schema, and migrates it to the latest version if necessary.
 * - Validates the workflow against the node templates, warning if the template is not known.
 * - Attempts to update nodes which have a mismatched version.
 * - Removes edges which are invalid.
 * - Reset image, board and model fields which are not accessible
 * @throws {z.ZodError} If there is a validation error.
 */
export const validateWorkflow = async (args: ValidateWorkflowArgs): Promise<ValidateWorkflowResult> => {
  const { workflow, templates, checkImageAccess, checkBoardAccess, checkModelAccess } = args;
  // Parse the raw workflow data & migrate it to the latest version
  const _workflow = parseAndMigrateWorkflow(workflow);

  // System workflows are only allowed to be used as templates.
  // If a system workflow is loaded, change its category to user and remove its ID so that we can save it as a user workflow.
  if (_workflow.meta.category === 'default') {
    _workflow.meta.category = 'user';
    _workflow.id = undefined;
  }

  // Now we can validate the graph
  const { nodes, edges } = _workflow;
  const warnings: WorkflowWarning[] = [];

  for (const node of nodes) {
    if (!isWorkflowInvocationNode(node)) {
      // We don't need to validate Note nodes or CurrentImage nodes - only Invocation nodes
      continue;
    }
    const template = templates[node.data.type];
    if (!template) {
      // This node's type template does not exist
      const message = t('nodes.missingTemplate', {
        node: node.id,
        type: node.data.type,
      });
      warnings.push({
        message,
        data: parseify(node),
      });
      continue;
    }

    // This node needs to be updated, based on comparison of its version to the template version
    if (getNeedsUpdate(node.data, template)) {
      try {
        const updatedNode = updateNode(node, template);
        node.data = updatedNode.data;
      } catch (e) {
        const message = t('nodes.unableToUpdateNode', {
          node: node.id,
          type: node.data.type,
        });
        warnings.push({
          message,
          data: parseify({ node, nodeTemplate: template }),
        });
        continue;
      }
    }

    for (const input of Object.values(node.data.inputs)) {
      const fieldTemplate = template.inputs[input.name];

      if (!fieldTemplate) {
        const message = t('nodes.missingFieldTemplate');
        warnings.push({
          message,
          data: parseify({ node, nodeTemplate: template, input }),
        });
        continue;
      }

      // We need to confirm that all images, boards and models are accessible before loading,
      // else the workflow could end up with stale data an an error state.
      if (fieldTemplate.type.name === 'ImageField' && input.value && isImageFieldInputInstance(input)) {
        const hasAccess = await checkImageAccess(input.value.image_name);
        if (!hasAccess) {
          const message = t('nodes.imageAccessError', { image_name: input.value.image_name });
          warnings.push({ message, data: parseify({ node, nodeTemplate: template, input }) });
          input.value = undefined;
        }
      }
      if (fieldTemplate.type.name === 'ImageField' && input.value && isImageFieldCollectionInputInstance(input)) {
        for (const { image_name } of [...input.value]) {
          const hasAccess = await checkImageAccess(image_name);
          if (!hasAccess) {
            const message = t('nodes.imageAccessError', { image_name });
            warnings.push({ message, data: parseify({ node, nodeTemplate: template, input }) });
            input.value = input.value.filter((image) => image.image_name !== image_name);
          }
        }
      }
      if (fieldTemplate.type.name === 'BoardField' && input.value && isBoardFieldInputInstance(input)) {
        const hasAccess = await checkBoardAccess(input.value.board_id);
        if (!hasAccess) {
          const message = t('nodes.boardAccessError', { board_id: input.value.board_id });
          warnings.push({ message, data: parseify({ node, nodeTemplate: template, input }) });
          input.value = undefined;
        }
      }
      if (isModelFieldType(fieldTemplate.type) && input.value && isModelIdentifierFieldInputInstance(input)) {
        const hasAccess = await checkModelAccess(input.value.key);
        if (!hasAccess) {
          const message = t('nodes.modelAccessError', { key: input.value.key });
          warnings.push({ message, data: parseify({ node, nodeTemplate: template, input }) });
          input.value = undefined;
        }
      }
    }
  }
  edges.forEach((edge, i) => {
    // Validate each edge. If the edge is invalid, we must remove it to prevent runtime errors with reactflow.
    const sourceNode = nodes.find(({ id }) => id === edge.source);
    const targetNode = nodes.find(({ id }) => id === edge.target);
    const sourceTemplate = sourceNode ? templates[sourceNode.data.type] : undefined;
    const targetTemplate = targetNode ? templates[targetNode.data.type] : undefined;
    const issues: string[] = [];

    if (!sourceNode) {
      // The edge's source/output node does not exist
      issues.push(
        t('nodes.sourceNodeDoesNotExist', {
          node: edge.source,
        })
      );
    }

    if (!sourceTemplate) {
      // The edge's source/output node template does not exist
      issues.push(
        t('nodes.missingTemplate', {
          node: edge.source,
          type: sourceNode?.data.type,
        })
      );
    }

    if (sourceNode && sourceTemplate && edge.type === 'default' && !(edge.sourceHandle in sourceTemplate.outputs)) {
      // The edge's source/output node field does not exist
      issues.push(
        t('nodes.sourceNodeFieldDoesNotExist', {
          node: edge.source,
          field: edge.sourceHandle,
        })
      );
    }

    if (!targetNode) {
      // The edge's target/input node does not exist
      issues.push(
        t('nodes.targetNodeDoesNotExist', {
          node: edge.target,
        })
      );
    }

    if (!targetTemplate) {
      // The edge's target/input node template does not exist
      issues.push(
        t('nodes.missingTemplate', {
          node: edge.target,
          type: targetNode?.data.type,
        })
      );
    }

    if (targetNode && targetTemplate && edge.type === 'default' && !(edge.targetHandle in targetTemplate.inputs)) {
      // The edge's target/input node field does not exist
      issues.push(
        t('nodes.targetNodeFieldDoesNotExist', {
          node: edge.target,
          field: edge.targetHandle,
        })
      );
    }

    if (issues.length) {
      // This edge has some issues. Remove it.
      delete edges[i];
      const source = edge.type === 'default' ? `${edge.source}.${edge.sourceHandle}` : edge.source;
      const target = edge.type === 'default' ? `${edge.source}.${edge.targetHandle}` : edge.target;
      warnings.push({
        message: t('nodes.deletedInvalidEdge', { source, target }),
        issues,
        data: edge,
      });
    }
  });

  // Migrated exposed fields to form elements if they exist and the form does not
  // Note: If the form is invalid per its zod schema, it will be reset to a default, empty form!
  if (_workflow.exposedFields.length > 0 && getIsFormEmpty(_workflow.form)) {
    _workflow.form = getDefaultForm();

    // Reverse the fields so that we can insert each at index 0 without needing to shift the index
    const reverseExposedFields = [..._workflow.exposedFields].reverse();

    for (const { nodeId, fieldName } of reverseExposedFields) {
      const node = nodes.find(({ id }) => id === nodeId);
      if (!node) {
        continue;
      }
      const nodeTemplate = templates[node.data.type];
      if (!nodeTemplate) {
        continue;
      }
      const fieldTemplate = nodeTemplate.inputs[fieldName];
      if (!fieldTemplate) {
        continue;
      }
      const element = buildNodeFieldElement(nodeId, fieldName, fieldTemplate.type);
      element.data.showDescription = false;
      addElement({
        form: _workflow.form,
        element,
        parentId: _workflow.form.rootElementId,
        index: 0,
      });
    }
  }

  // If the form exists, remove any form elements that no longer have a corresponding node field
  for (const element of Object.values(_workflow.form.elements)) {
    if (!isNodeFieldElement(element)) {
      continue;
    }
    const { nodeId, fieldName } = element.data.fieldIdentifier;
    const node = nodes.filter(isWorkflowInvocationNode).find(({ id }) => id === nodeId);
    const field = node?.data.inputs[fieldName];
    if (!field) {
      // The form element field no longer exists on the node
      delete _workflow.form.elements[element.id];
      warnings.push({
        message: t('nodes.deletedMissingNodeFieldFormElement', { nodeId, fieldName }),
        data: { nodeId, fieldName },
      });
    }
  }

  return { workflow: _workflow, warnings };
};
