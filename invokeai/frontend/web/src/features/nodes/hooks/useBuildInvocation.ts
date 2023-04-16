import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';
import { reduce } from 'lodash';
import { Node } from 'reactflow';
import { v4 as uuidv4 } from 'uuid';
import { InputFieldValue, InvocationValue, OutputFieldValue } from '../types';
import { buildInputFieldValue } from '../util/fieldValueBuilders';

export const useBuildInvocation = () => {
  const invocationTemplates = useAppSelector(
    (state: RootState) => state.nodes.invocationTemplates
  );

  return (type: string) => {
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

    const invocation: Node<InvocationValue> = {
      id: nodeId,
      type: 'invocation',
      position: { x: 0, y: 0 },
      data: {
        id: nodeId,
        type,
        inputs,
        outputs,
      },
    };

    return invocation;
  };
};
