import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { satisfies } from 'compare-versions';
import { useCallback, useMemo } from 'react';
import {
  InvocationNodeData,
  isInvocationNode,
  zParsedSemver,
} from '../types/types';
import { cloneDeep, defaultsDeep } from 'lodash-es';
import { buildNodeData } from '../store/util/buildNodeData';
import { AnyInvocationType } from 'services/events/types';
import { Node } from 'reactflow';
import { nodeReplaced } from '../store/nodesSlice';

export const useNodeVersion = (nodeId: string) => {
  const dispatch = useAppDispatch();
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          const nodeTemplate = nodes.nodeTemplates[node?.data.type ?? ''];
          return { node, nodeTemplate };
        },
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const { node, nodeTemplate } = useAppSelector(selector);

  const needsUpdate = useMemo(() => {
    if (!isInvocationNode(node) || !nodeTemplate) {
      return false;
    }
    return node.data.version !== nodeTemplate.version;
  }, [node, nodeTemplate]);

  const mayUpdate = useMemo(() => {
    if (
      !needsUpdate ||
      !isInvocationNode(node) ||
      !nodeTemplate ||
      !node.data.version
    ) {
      return false;
    }
    const templateMajor = zParsedSemver.parse(nodeTemplate.version).major;

    return satisfies(node.data.version, `^${templateMajor}`);
  }, [needsUpdate, node, nodeTemplate]);

  const updateNode = useCallback(() => {
    if (
      !mayUpdate ||
      !isInvocationNode(node) ||
      !nodeTemplate ||
      !node.data.version
    ) {
      return;
    }

    const defaults = buildNodeData(
      node.data.type as AnyInvocationType,
      node.position,
      nodeTemplate
    ) as Node<InvocationNodeData>;

    const clone = cloneDeep(node);
    clone.data.version = nodeTemplate.version;
    defaultsDeep(clone, defaults);
    dispatch(nodeReplaced({ nodeId: clone.id, node: clone }));
  }, [dispatch, mayUpdate, node, nodeTemplate]);

  return { needsUpdate, mayUpdate, updateNode };
};
