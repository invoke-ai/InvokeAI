import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useNodeTemplateTitleOrThrow = (nodeId: string): string => {
  const templates = useStore($templates);
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodesSlice) => {
        const node = nodesSlice.nodes.find((node) => node.id === nodeId);
        assert(isInvocationNode(node), 'Node not found');
        const template = templates[node.data.type];
        assert(template, 'Template not found');
        return template.title;
      }),
    [nodeId, templates]
  );
  const title = useAppSelector(selector);
  return title;
};
