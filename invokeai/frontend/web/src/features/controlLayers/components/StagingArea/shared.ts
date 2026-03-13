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

export const getQueueItemElementId = (index: number) => `queue-item-preview-${index}`;

export const getOutputImageNames = (item: S['SessionQueueItem']): string[] => {
  const imageNames: string[] = [];

  for (const [sourceNodeId, preparedNodeIds] of Object.entries(item.session.source_prepared_mapping)) {
    if (!isCanvasOutputNodeId(sourceNodeId)) {
      continue;
    }
    const nodeId = preparedNodeIds[0];
    const output = nodeId ? item.session.results[nodeId] : undefined;
    if (!output) {
      continue;
    }
    for (const [_name, value] of objectEntries(output)) {
      if (isImageField(value)) {
        imageNames.push(value.image_name);
      }
    }
  }

  return imageNames;
};
