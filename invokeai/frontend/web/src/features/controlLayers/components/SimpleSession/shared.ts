import { skipToken } from '@reduxjs/toolkit/query';
import { isImageField } from 'features/nodes/types/common';
import { isCanvasOutputNodeId } from 'features/nodes/util/graph/graphBuilderUtils';
import { round } from 'lodash-es';
import { useMemo } from 'react';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
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

export const getQueueItemElementId = (item_id: number) => `queue-item-status-card-${item_id}`;

const getOutputImageName = (item: S['SessionQueueItem']) => {
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
  const outputImageName = useMemo(() => getOutputImageName(item), [item]);

  const { currentData: imageDTO } = useGetImageDTOQuery(outputImageName ?? skipToken);

  return imageDTO;
};
