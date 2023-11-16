import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { satisfies } from 'compare-versions';
import { cloneDeep, defaultsDeep } from 'lodash-es';
import { useCallback, useMemo } from 'react';
import { Node } from 'reactflow';
import { AnyInvocationType } from 'services/events/types';
import { nodeReplaced } from '../store/nodesSlice';
import { buildNodeData } from '../store/util/buildNodeData';
import {
  InvocationNodeData,
  InvocationTemplate,
  NodeData,
  isInvocationNode,
  zParsedSemver,
} from '../types/types';
import { useAppToaster } from 'app/components/Toaster';
import { useTranslation } from 'react-i18next';

export const getNeedsUpdate = (
  node?: Node<NodeData>,
  template?: InvocationTemplate
) => {
  if (!isInvocationNode(node) || !template) {
    return false;
  }
  return node.data.version !== template.version;
};

export const getMayUpdateNode = (
  node?: Node<NodeData>,
  template?: InvocationTemplate
) => {
  const needsUpdate = getNeedsUpdate(node, template);
  if (
    !needsUpdate ||
    !isInvocationNode(node) ||
    !template ||
    !node.data.version
  ) {
    return false;
  }
  const templateMajor = zParsedSemver.parse(template.version).major;

  return satisfies(node.data.version, `^${templateMajor}`);
};

export const updateNode = (
  node?: Node<NodeData>,
  template?: InvocationTemplate
) => {
  const mayUpdate = getMayUpdateNode(node, template);
  if (
    !mayUpdate ||
    !isInvocationNode(node) ||
    !template ||
    !node.data.version
  ) {
    return;
  }

  const defaults = buildNodeData(
    node.data.type as AnyInvocationType,
    node.position,
    template
  ) as Node<InvocationNodeData>;

  const clone = cloneDeep(node);
  clone.data.version = template.version;
  defaultsDeep(clone, defaults);
  return clone;
};

export const useNodeVersion = (nodeId: string) => {
  const dispatch = useAppDispatch();
  const toast = useAppToaster();
  const { t } = useTranslation();
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

  const needsUpdate = useMemo(
    () => getNeedsUpdate(node, nodeTemplate),
    [node, nodeTemplate]
  );

  const mayUpdate = useMemo(
    () => getMayUpdateNode(node, nodeTemplate),
    [node, nodeTemplate]
  );

  const _updateNode = useCallback(() => {
    const needsUpdate = getNeedsUpdate(node, nodeTemplate);
    const updatedNode = updateNode(node, nodeTemplate);
    if (!updatedNode) {
      if (needsUpdate) {
        toast({ title: t('nodes.unableToUpdateNodes', { count: 1 }) });
      }
      return;
    }
    dispatch(nodeReplaced({ nodeId: updatedNode.id, node: updatedNode }));
  }, [dispatch, node, nodeTemplate, t, toast]);

  return { needsUpdate, mayUpdate, updateNode: _updateNode };
};
