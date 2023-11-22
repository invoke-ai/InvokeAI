import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import {
  CurrentImageNodeData,
  InputFieldValue,
  InvocationNodeData,
  InvocationTemplate,
  NotesNodeData,
  OutputFieldValue,
} from 'features/nodes/types/types';
import { buildInputFieldValue } from 'features/nodes/util/fieldValueBuilders';
import { reduce } from 'lodash-es';
import { Node, XYPosition } from 'reactflow';
import { AnyInvocationType } from 'services/events/types';
import { v4 as uuidv4 } from 'uuid';

export const SHARED_NODE_PROPERTIES: Partial<Node> = {
  dragHandle: `.${DRAG_HANDLE_CLASSNAME}`,
};
export const buildNodeData = (
  type: AnyInvocationType | 'current_image' | 'notes',
  position: XYPosition,
  template?: InvocationTemplate
):
  | Node<CurrentImageNodeData>
  | Node<NotesNodeData>
  | Node<InvocationNodeData>
  | undefined => {
  const nodeId = uuidv4();

  if (type === 'current_image') {
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
  }

  if (type === 'notes') {
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
  }

  if (template === undefined) {
    console.error(`Unable to find template ${type}.`);
    return;
  }

  const inputs = reduce(
    template.inputs,
    (inputsAccumulator, inputTemplate, inputName) => {
      const fieldId = uuidv4();

      const inputFieldValue: InputFieldValue = buildInputFieldValue(
        fieldId,
        inputTemplate
      );

      inputsAccumulator[inputName] = inputFieldValue;

      return inputsAccumulator;
    },
    {} as Record<string, InputFieldValue>
  );

  const outputs = reduce(
    template.outputs,
    (outputsAccumulator, outputTemplate, outputName) => {
      const fieldId = uuidv4();

      const outputFieldValue: OutputFieldValue = {
        id: fieldId,
        name: outputName,
        type: outputTemplate.type,
        fieldKind: 'output',
      };

      outputsAccumulator[outputName] = outputFieldValue;

      return outputsAccumulator;
    },
    {} as Record<string, OutputFieldValue>
  );

  const invocation: Node<InvocationNodeData> = {
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
      inputs,
      outputs,
      useCache: template.useCache,
    },
  };

  return invocation;
};
