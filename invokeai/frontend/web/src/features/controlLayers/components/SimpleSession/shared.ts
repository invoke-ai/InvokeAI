import { useStore } from '@nanostores/react';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { isImageField } from 'features/nodes/types/common';
import { isCanvasOutputNodeId } from 'features/nodes/util/graph/graphBuilderUtils';
import { round } from 'lodash-es';
import { computed } from 'nanostores';
import { useState } from 'react';
import type { S } from 'services/api/types';
import { objectEntries } from 'tsafe';

export const getProgressMessage = (data?: S['InvocationProgressEvent'] | null) => {
  if (!data) {
    return 'Generating';
  }

  let message = data.message;
  if (data.percentage) {
    message += ` (${round(data.percentage * 100)}%)`;
  }
  return message;
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

export const useOutputImageDTO = (item: S['SessionQueueItem']) => {
  const ctx = useCanvasSessionContext();
  const $imageDTO = useState(() =>
    computed([ctx.$progressData], (progressData) => progressData[item.item_id]?.imageDTO ?? null)
  )[0];
  const imageDTO = useStore($imageDTO);

  return imageDTO;
};
