import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { map } from 'lodash-es';
import { useMemo } from 'react';
import { KIND_MAP } from '../types/constants';
import { isInvocationNode } from '../types/types';

export const useFieldNames = (nodeId: string, kind: 'input' | 'output') => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          if (!isInvocationNode(node)) {
            return [];
          }
          return map(node.data[KIND_MAP[kind]], (field) => field.name).filter(
            (fieldName) => fieldName !== 'is_intermediate'
          );
        },
        defaultSelectorOptions
      ),
    [kind, nodeId]
  );

  const fieldNames = useAppSelector(selector);
  return fieldNames;
};
