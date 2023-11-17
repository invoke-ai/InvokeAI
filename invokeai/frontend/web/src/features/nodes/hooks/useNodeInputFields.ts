import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useMemo } from 'react';
import {
  InputFieldTemplate,
  InputFieldValue,
  isInvocationNode,
} from '../types/types';

export const useNodeInputFields = (
  nodeId: string
): { fieldData: InputFieldValue; fieldTemplate: InputFieldTemplate }[] => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          if (!isInvocationNode(node)) {
            return [];
          }

          const template = nodes.nodeTemplates[node.data.type];

          if (!template) {
            return [];
          }

          const inputs = Object.values(node.data.inputs).reduce<
            {
              fieldData: InputFieldValue;
              fieldTemplate: InputFieldTemplate;
            }[]
          >((acc, fieldData) => {
            const fieldTemplate = template.inputs[fieldData.name];
            if (fieldTemplate) {
              acc.push({
                fieldData,
                fieldTemplate,
              });
            }
            return acc;
          }, []);

          return inputs;
        },
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const inputs = useAppSelector(selector);
  return inputs;
};
