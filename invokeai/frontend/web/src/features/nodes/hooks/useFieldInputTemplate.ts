import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useMemo } from 'react';
import { isInvocationNode } from 'features/nodes/types/invocation';

export const useFieldInputTemplate = (nodeId: string, fieldName: string) => {
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
          return nodeTemplate?.inputs[fieldName];
        },
        defaultSelectorOptions
      ),
    [fieldName, nodeId]
  );

  const fieldTemplate = useAppSelector(selector);

  return fieldTemplate;
};
