import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { updateAllNodesRequested } from 'features/nodes/store/actions';
import { nodeReplaced } from 'features/nodes/store/nodesSlice';
import { NodeUpdateError } from 'features/nodes/types/error';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { getNeedsUpdate, updateNode } from 'features/nodes/util/node/nodeUpdate';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { t } from 'i18next';

export const addUpdateAllNodesRequestedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: updateAllNodesRequested,
    effect: (action, { dispatch, getState }) => {
      const log = logger('nodes');
      const { nodes, templates } = getState().nodes;

      let unableToUpdateCount = 0;

      nodes.filter(isInvocationNode).forEach((node) => {
        const template = templates[node.data.type];
        if (!template) {
          unableToUpdateCount++;
          return;
        }
        if (!getNeedsUpdate(node, template)) {
          // No need to increment the count here, since we're not actually updating
          return;
        }
        try {
          const updatedNode = updateNode(node, template);
          dispatch(nodeReplaced({ nodeId: updatedNode.id, node: updatedNode }));
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
        dispatch(
          addToast(
            makeToast({
              title: t('nodes.unableToUpdateNodes', {
                count: unableToUpdateCount,
              }),
            })
          )
        );
      } else {
        dispatch(
          addToast(
            makeToast({
              title: t('nodes.allNodesUpdated'),
              status: 'success',
            })
          )
        );
      }
    },
  });
};
