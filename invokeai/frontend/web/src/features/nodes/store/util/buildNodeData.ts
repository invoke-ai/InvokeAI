import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import {
  FieldInputInstance,
  FieldOutputInstance,
} from 'features/nodes/types/field';
import {
  CurrentImageNodeData,
  InvocationNodeData,
  InvocationTemplate,
  NotesNodeData,
} from 'features/nodes/types/invocation';
import { buildFieldInputInstance } from 'features/nodes/util/buildFieldInputInstance';
import { reduce } from 'lodash-es';
import { Node, XYPosition } from 'reactflow';
import { v4 as uuidv4 } from 'uuid';

export const SHARED_NODE_PROPERTIES: Partial<Node> = {
  dragHandle: `.${DRAG_HANDLE_CLASSNAME}`,
};

export const buildNotesNode = (position: XYPosition): Node<NotesNodeData> => {
  const nodeId = uuidv4();
  const node: Node<NotesNodeData> = {
    ...SHARED_NODE_PROPERTIES,
    id: nodeId,
    type: 'notes',
    position,
    data: {
      id: nodeId,
      isOpen: true,
      label: 'Notes',
      notes: '',
      type: 'notes',
    },
  };
  return node;
};

export const buildCurrentImageNode = (
  position: XYPosition
): Node<CurrentImageNodeData> => {
  const nodeId = uuidv4();
  const node: Node<CurrentImageNodeData> = {
    ...SHARED_NODE_PROPERTIES,
    id: nodeId,
    type: 'current_image',
    position,
    data: {
      id: nodeId,
      type: 'current_image',
      isOpen: true,
      label: 'Current Image',
    },
  };
  return node;
};

export const buildInvocationNode = (
  position: XYPosition,
  template: InvocationTemplate
): Node<InvocationNodeData> => {
  const nodeId = uuidv4();
  const { type } = template;

  const inputs = reduce(
    template.inputs,
    (inputsAccumulator, inputTemplate, inputName) => {
      const fieldId = uuidv4();

      const inputFieldValue: FieldInputInstance = buildFieldInputInstance(
        fieldId,
        inputTemplate
      );

      inputsAccumulator[inputName] = inputFieldValue;

      return inputsAccumulator;
    },
    {} as Record<string, FieldInputInstance>
  );

  const outputs = reduce(
    template.outputs,
    (outputsAccumulator, outputTemplate, outputName) => {
      const fieldId = uuidv4();

      const outputFieldValue: FieldOutputInstance = {
        id: fieldId,
        name: outputName,
        type: outputTemplate.type,
        fieldKind: 'output',
      };

      outputsAccumulator[outputName] = outputFieldValue;

      return outputsAccumulator;
    },
    {} as Record<string, FieldOutputInstance>
  );

  const node: Node<InvocationNodeData> = {
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
      embedWorkflow: false,
      isIntermediate: type === 'save_image' ? false : true,
      useCache: template.useCache,
      inputs,
      outputs,
    },
  };

  return node;
};
