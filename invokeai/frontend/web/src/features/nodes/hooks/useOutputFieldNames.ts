import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { map } from 'lodash-es';
import { useMemo } from 'react';
import { isInvocationNode } from '../types/invocation';
import { getSortedFilteredFieldNames } from '../util/getSortedFilteredFieldNames';

export const useOutputFieldNames = (nodeId: string) => {
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

          return getSortedFilteredFieldNames(map(nodeTemplate.outputs));
        },
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const fieldNames = useAppSelector(selector);
  return fieldNames;
};
