import { createSelector } from '@reduxjs/toolkit';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectNodeTemplate } from 'features/nodes/store/selectors';
import { getSortedFilteredFieldNames } from 'features/nodes/util/node/getSortedFilteredFieldNames';
import { map } from 'lodash-es';
import { useMemo } from 'react';

export const useOutputFieldNames = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        const template = selectNodeTemplate(nodes, nodeId);
        if (!template) {
          return EMPTY_ARRAY;
        }

        return getSortedFilteredFieldNames(map(template.outputs));
      }),
    [nodeId]
  );

  const fieldNames = useAppSelector(selector);
  return fieldNames;
};
