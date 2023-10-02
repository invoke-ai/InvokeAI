import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useMemo } from 'react';

export const useNodeTemplate = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.present.nodes.find((node) => node.id === nodeId);
          const nodeTemplate = nodes.present.nodeTemplates[node?.data.type ?? ''];
          return nodeTemplate;
        },
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const nodeTemplate = useAppSelector(selector);

  return nodeTemplate;
};
