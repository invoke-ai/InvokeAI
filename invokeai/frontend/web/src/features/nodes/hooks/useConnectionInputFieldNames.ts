import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { map } from 'lodash-es';
import { useMemo } from 'react';
import { isInvocationNode } from '../types/types';

export const useConnectionInputFieldNames = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          if (!isInvocationNode(node)) {
            return [];
          }
          const nodeTemplate = nodes.nodeTemplates[node.data.type];
          if (!nodeTemplate) {
            return [];
          }
          return map(nodeTemplate.inputs)
            .filter((field) => field.input === 'connection')
            .filter((field) => !field.ui_hidden)
            .sort((a, b) => (a.ui_order ?? 0) - (b.ui_order ?? 0))
            .map((field) => field.name)
            .filter((fieldName) => fieldName !== 'is_intermediate');
        },
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const fieldNames = useAppSelector(selector);
  return fieldNames;
};
