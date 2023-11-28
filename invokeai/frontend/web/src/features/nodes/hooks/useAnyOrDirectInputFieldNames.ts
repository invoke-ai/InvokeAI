import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { keys, map } from 'lodash-es';
import { useMemo } from 'react';
import { isInvocationNode } from '../types/invocation';
import { getSortedFilteredFieldNames } from '../util/node/getSortedFilteredFieldNames';
import { TEMPLATE_BUILDER_MAP } from '../util/schema/buildFieldInputTemplate';

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
                field.type.isCollectionOrScalar) &&
              keys(TEMPLATE_BUILDER_MAP).includes(field.type.name)
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
