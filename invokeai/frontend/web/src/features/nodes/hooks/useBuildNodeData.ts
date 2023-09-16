import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { reduce } from 'lodash-es';
import { useCallback } from 'react';
import { Node, useReactFlow } from 'reactflow';
import { AnyInvocationType } from 'services/events/types';
import { v4 as uuidv4 } from 'uuid';
import {
  CurrentImageNodeData,
  InputFieldValue,
  InvocationNodeData,
  NotesNodeData,
  OutputFieldValue,
} from '../types/types';
import { buildInputFieldValue } from '../util/fieldValueBuilders';
import { DRAG_HANDLE_CLASSNAME, NODE_WIDTH } from '../types/constants';

const templatesSelector = createSelector(
  [(state: RootState) => state.nodes],
  (nodes) => nodes.nodeTemplates
);

export const SHARED_NODE_PROPERTIES: Partial<Node> = {
  dragHandle: `.${DRAG_HANDLE_CLASSNAME}`,
};

export const useBuildNodeData = () => {
  const invocationTemplates = useAppSelector(templatesSelector);

  const flow = useReactFlow();

  return useCallback(
    (type: AnyInvocationType | 'current_image' | 'notes') => {
      const nodeId = uuidv4();

      let _x = window.innerWidth / 2;
      let _y = window.innerHeight / 2;

      // attempt to center the node in the middle of the flow
      const rect = document
        .querySelector('#workflow-editor')
        ?.getBoundingClientRect();

      if (rect) {
        _x = rect.width / 2 - NODE_WIDTH / 2;
        _y = rect.height / 2 - NODE_WIDTH / 2;
      }

      const { x, y } = flow.project({
        x: _x,
        y: _y,
      });

      if (type === 'current_image') {
        const node: Node<CurrentImageNodeData> = {
          ...SHARED_NODE_PROPERTIES,
          id: nodeId,
          type: 'current_image',
          position: { x: x, y: y },
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
          position: { x: x, y: y },
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

      const template = invocationTemplates[type];

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
        position: { x: x, y: y },
        data: {
          id: nodeId,
          type,
          version: template.version,
          label: '',
          notes: '',
          isOpen: true,
          embedWorkflow: false,
          isIntermediate: true,
          inputs,
          outputs,
        },
      };

      return invocation;
    },
    [invocationTemplates, flow]
  );
};
