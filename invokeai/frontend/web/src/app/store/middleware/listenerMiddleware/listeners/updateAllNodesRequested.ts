import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { updateAllNodesRequested } from 'features/nodes/store/actions';
import { $templates, nodesChanged } from 'features/nodes/store/nodesSlice';
import { selectNodes } from 'features/nodes/store/selectors';
import { NodeUpdateError } from 'features/nodes/types/error';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { getNeedsUpdate, updateNode } from 'features/nodes/util/node/nodeUpdate';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';

const log = logger('workflows');

export const addUpdateAllNodesRequestedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: updateAllNodesRequested,
    effect: (action, { dispatch, getState }) => {
      const nodes = selectNodes(getState());
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
          dispatch(
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
    },
  });
};
