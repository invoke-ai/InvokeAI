import { IconButton } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { useAppStore } from 'app/store/storeHooks';
import { useGetNodesNeedUpdate } from 'features/nodes/hooks/useGetNodesNeedUpdate';
import { $templates, nodesChanged } from 'features/nodes/store/nodesSlice';
import { selectNodes } from 'features/nodes/store/selectors';
import { NodeUpdateError } from 'features/nodes/types/error';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { getNeedsUpdate, updateNode } from 'features/nodes/util/node/nodeUpdate';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiWarningBold } from 'react-icons/pi';

const log = logger('workflows');

const useUpdateNodes = () => {
  const store = useAppStore();
  const { t } = useTranslation();

  const updateNodes = useCallback(() => {
    const nodes = selectNodes(store.getState());
    const templates = $templates.get();

    let unableToUpdateCount = 0;

    nodes.filter(isInvocationNode).forEach((node) => {
      const template = templates[node.data.type];
      if (!template) {
        unableToUpdateCount++;
        return;
      }
      if (!getNeedsUpdate(node.data, template)) {
        // No need to increment the count here, since we're not actually updating
        return;
      }
      try {
        const updatedNode = updateNode(node, template);
        store.dispatch(
          nodesChanged([
            { type: 'remove', id: updatedNode.id },
            { type: 'add', item: updatedNode },
          ])
        );
      } catch (e) {
        if (e instanceof NodeUpdateError) {
          unableToUpdateCount++;
        }
      }
    });

    if (unableToUpdateCount) {
      log.warn(
        t('nodes.unableToUpdateNodes', {
          count: unableToUpdateCount,
        })
      );
      toast({
        id: 'UNABLE_TO_UPDATE_NODES',
        title: t('nodes.unableToUpdateNodes', {
          count: unableToUpdateCount,
        }),
      });
    } else {
      toast({
        id: 'ALL_NODES_UPDATED',
        title: t('nodes.allNodesUpdated'),
        status: 'success',
      });
    }
  }, [store, t]);

  return updateNodes;
};

const UpdateNodesButton = () => {
  const { t } = useTranslation();
  const nodesNeedUpdate = useGetNodesNeedUpdate();
  const updateNodes = useUpdateNodes();

  if (!nodesNeedUpdate) {
    return null;
  }

  return (
    <IconButton
      tooltip={t('nodes.updateAllNodes')}
      aria-label={t('nodes.updateAllNodes')}
      icon={<PiWarningBold />}
      onClick={updateNodes}
      pointerEvents="auto"
      colorScheme="warning"
    />
  );
};

export default memo(UpdateNodesButton);
