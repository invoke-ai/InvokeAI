import {
  getNeedsUpdate,
  updateNode,
} from 'features/nodes/hooks/useNodeVersion';
import { updateAllNodesRequested } from 'features/nodes/store/actions';
import { nodeReplaced } from 'features/nodes/store/nodesSlice';
import { startAppListening } from '..';
import { logger } from 'app/logging/logger';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { t } from 'i18next';

export const addUpdateAllNodesRequestedListener = () => {
  startAppListening({
    actionCreator: updateAllNodesRequested,
    effect: (action, { dispatch, getState }) => {
      const log = logger('nodes');
      const nodes = getState().nodes.nodes;
      const templates = getState().nodes.nodeTemplates;

      let unableToUpdateCount = 0;

      nodes.forEach((node) => {
        const template = templates[node.data.type];
        const needsUpdate = getNeedsUpdate(node, template);
        const updatedNode = updateNode(node, template);
        if (!updatedNode) {
          if (needsUpdate) {
            unableToUpdateCount++;
          }
          return;
        }
        dispatch(nodeReplaced({ nodeId: updatedNode.id, node: updatedNode }));
      });

      if (unableToUpdateCount) {
        log.warn(
          `Unable to update ${unableToUpdateCount} nodes. Please report this issue.`
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
      }
    },
  });
};
