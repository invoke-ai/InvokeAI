import { SHARED_NODE_PROPERTIES } from 'features/nodes/types/constants';
import type { FieldInputInstance } from 'features/nodes/types/field';
import type { InvocationNode, InvocationTemplate } from 'features/nodes/types/invocation';
import { buildFieldInputInstance } from 'features/nodes/util/schema/buildFieldInputInstance';
import { reduce } from 'lodash-es';
import type { XYPosition } from 'reactflow';
import { v4 as uuidv4 } from 'uuid';

export const buildInvocationNode = (position: XYPosition, template: InvocationTemplate): InvocationNode => {
  const nodeId = uuidv4();
  const { type } = template;

  const inputs = reduce(
    template.inputs,
    (inputsAccumulator, inputTemplate, inputName) => {
      const fieldId = uuidv4();

      const inputFieldValue: FieldInputInstance = buildFieldInputInstance(fieldId, inputTemplate);

      inputsAccumulator[inputName] = inputFieldValue;

      return inputsAccumulator;
    },
    {} as Record<string, FieldInputInstance>
  );

  const node: InvocationNode = {
    ...SHARED_NODE_PROPERTIES,
    id: nodeId,
    type: 'invocation',
    position,
    data: {
      id: nodeId,
      type,
      version: template.version,
      label: '',
      notes: '',
      isOpen: true,
      isIntermediate: type === 'save_image' ? false : true,
      useCache: template.useCache,
      inputs,
    },
  };

  return node;
};
