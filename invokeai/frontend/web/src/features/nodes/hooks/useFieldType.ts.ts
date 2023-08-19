import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useMemo } from 'react';
import { KIND_MAP } from '../types/constants';
import { isInvocationNode } from '../types/types';

export const useFieldType = (
  nodeId: string,
  fieldName: string,
  kind: 'input' | 'output'
) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          if (!isInvocationNode(node)) {
            return;
          }
          return node?.data[KIND_MAP[kind]][fieldName]?.type;
        },
        defaultSelectorOptions
      ),
    [fieldName, kind, nodeId]
  );

  const fieldType = useAppSelector(selector);

  return fieldType;
};
