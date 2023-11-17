import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useMemo } from 'react';
import { isInvocationNode } from '../types/invocation';
import { getNeedsUpdate } from '../store/util/nodeUpdate';

export const useNodeNeedsUpdate = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          const template = nodes.nodeTemplates[node?.data.type ?? ''];
          return { node, template };
        },
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const { node, template } = useAppSelector(selector);

  const needsUpdate = useMemo(
    () =>
      isInvocationNode(node) && template
        ? getNeedsUpdate(node, template)
        : false,
    [node, template]
  );

  return needsUpdate;
};
