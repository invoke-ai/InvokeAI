import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { compareVersions } from 'compare-versions';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useDoNodeVersionsMatch = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(stateSelector, ({ nodes }) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        if (!isInvocationNode(node)) {
          return false;
        }
        const nodeTemplate = nodes.nodeTemplates[node?.data.type ?? ''];
        if (!nodeTemplate?.version || !node.data?.version) {
          return false;
        }
        return compareVersions(nodeTemplate.version, node.data.version) === 0;
      }),
    [nodeId]
  );

  const nodeTemplate = useAppSelector(selector);

  return nodeTemplate;
};
