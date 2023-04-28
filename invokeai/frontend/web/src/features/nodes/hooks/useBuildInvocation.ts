import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { reduce } from 'lodash-es';
import { useCallback } from 'react';
import { Node, useReactFlow } from 'reactflow';
import { AnyInvocationType } from 'services/events/types';
import { v4 as uuidv4 } from 'uuid';
import {
  InputFieldValue,
  InvocationValue,
  OutputFieldValue,
} from '../types/types';
import { buildInputFieldValue } from '../util/fieldValueBuilders';

const templatesSelector = createSelector(
  [(state: RootState) => state.nodes],
  (nodes) => nodes.invocationTemplates
);

export const useBuildInvocation = () => {
  const invocationTemplates = useAppSelector(templatesSelector);

  const flow = useReactFlow();

  return useCallback(
    (type: AnyInvocationType) => {
      const template = invocationTemplates[type];

      if (template === undefined) {
        console.error(`Unable to find template ${type}.`);
        return;
      }

      const nodeId = uuidv4();

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
          };

          outputsAccumulator[outputName] = outputFieldValue;

          return outputsAccumulator;
        },
        {} as Record<string, OutputFieldValue>
      );

      const { x, y } = flow.project({
        x: window.innerWidth / 2.5,
        y: window.innerHeight / 8,
      });

      const invocation: Node<InvocationValue> = {
        id: nodeId,
        type: 'invocation',
        position: { x: x, y: y },
        data: {
          id: nodeId,
          type,
          inputs,
          outputs,
        },
      };

      return invocation;
    },
    [invocationTemplates, flow]
  );
};
