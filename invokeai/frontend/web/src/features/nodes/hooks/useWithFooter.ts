import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { some } from 'lodash-es';
import { useMemo } from 'react';
import { FOOTER_FIELDS } from '../types/constants';
import { isInvocationNode } from '../types/types';

export const useWithFooter = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          if (!isInvocationNode(node)) {
            return false;
          }
          return some(node.data.outputs, (output) =>
            FOOTER_FIELDS.includes(output.type)
          );
        },
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const withFooter = useAppSelector(selector);
  return withFooter;
};
