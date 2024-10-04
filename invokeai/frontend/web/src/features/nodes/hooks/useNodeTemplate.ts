import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectInvocationNodeType, selectNodesSlice } from 'features/nodes/store/selectors';
import type { InvocationTemplate } from 'features/nodes/types/invocation';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useNodeTemplate = (nodeId: string): InvocationTemplate => {
  const templates = useStore($templates);
  const selectNodeType = useMemo(
    () => createSelector(selectNodesSlice, (nodes) => selectInvocationNodeType(nodes, nodeId)),
    [nodeId]
  );
  const nodeType = useAppSelector(selectNodeType);
  const template = useMemo(() => {
    const t = templates[nodeType];
    assert(t, `Template for node type ${nodeType} not found`);
    return t;
  }, [nodeType, templates]);
  return template;
};
