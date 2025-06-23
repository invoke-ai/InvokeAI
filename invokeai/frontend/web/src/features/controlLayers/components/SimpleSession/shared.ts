import { isImageField } from 'features/nodes/types/common';
import { isCanvasOutputNodeId } from 'features/nodes/util/graph/graphBuilderUtils';
import type { S } from 'services/api/types';
import { formatProgressMessage } from 'services/events/stores';
import { objectEntries } from 'tsafe';

export const getProgressMessage = (data?: S['InvocationProgressEvent'] | null) => {
  if (!data) {
    return 'Generating';
  }
  return formatProgressMessage(data);
};

export const DROP_SHADOW = 'drop-shadow(0px 0px 4px rgb(0, 0, 0)) drop-shadow(0px 0px 4px rgba(0, 0, 0, 0.3))';

export const getQueueItemElementId = (itemId: number) => `queue-item-status-card-${itemId}`;

export const getOutputImageName = (item: S['SessionQueueItem']) => {
  const nodeId = Object.entries(item.session.source_prepared_mapping).find(([nodeId]) =>
    isCanvasOutputNodeId(nodeId)
  )?.[1][0];
  const output = nodeId ? item.session.results[nodeId] : undefined;

  if (!output) {
    return null;
  }

  for (const [_name, value] of objectEntries(output)) {
    if (isImageField(value)) {
      return value.image_name;
    }
  }

  return null;
};
