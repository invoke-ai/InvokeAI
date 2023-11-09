import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useMemo } from 'react';
import { isInvocationNode } from '../types/types';

export const useFieldInputKind = (nodeId: string, fieldName: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          if (!isInvocationNode(node)) {
            return;
          }
          const nodeTemplate = nodes.nodeTemplates[node?.data.type ?? ''];
          const fieldTemplate = nodeTemplate?.inputs[fieldName];
          return fieldTemplate?.input;
        },
        defaultSelectorOptions
      ),
    [fieldName, nodeId]
  );

  const inputKind = useAppSelector(selector);

  return inputKind;
};
