import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { map } from 'lodash-es';
import { useMemo } from 'react';
import { isInvocationNode } from '../types/types';
import {
  POLYMORPHIC_TYPES,
  TYPES_WITH_INPUT_COMPONENTS,
} from '../types/constants';
import { getSortedFilteredFieldNames } from '../util/getSortedFilteredFieldNames';

export const useAnyOrDirectInputFieldNames = (nodeId: string) => {
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
          const fields = map(nodeTemplate.inputs).filter(
            (field) =>
              (['any', 'direct'].includes(field.input) ||
                POLYMORPHIC_TYPES.includes(field.type)) &&
              TYPES_WITH_INPUT_COMPONENTS.includes(field.type)
          );
          return getSortedFilteredFieldNames(fields);
        },
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const fieldNames = useAppSelector(selector);
  return fieldNames;
};
