import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useMemo } from 'react';
import { isInvocationNode } from '../types/types';

export const useUseCache = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          if (!isInvocationNode(node)) {
            return false;
          }
          // cast to boolean to support older workflows that didn't have useCache
          // TODO: handle this better somehow
          return node.data.useCache;
        },
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const useCache = useAppSelector(selector);
  return useCache;
};
