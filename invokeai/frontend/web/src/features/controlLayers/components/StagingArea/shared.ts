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

export const getOutputImageName = (item: S['SessionQueueItem']) => {
  const nodeId = Object.entries(item.session.source_prepared_mapping).find(([nodeId]) =>
    isCanvasOutputNodeId(nodeId)
  )?.[1][0];
  const output = nodeId ? item.session.results[nodeId] : undefined;

  const getImageNameFromOutput = (result?: S['GraphExecutionState']['results'][string]) => {
    if (!result) {
      return null;
    }
    for (const [_name, value] of objectEntries(result)) {
      if (isImageField(value)) {
        return value.image_name;
      }
    }
    return null;
  };

  const imageName = getImageNameFromOutput(output);
  if (imageName) {
    return imageName;
  }

  // Fallback: search all results for an image field. Custom workflows may not have a canvas_output-prefixed node id.
  for (const result of Object.values(item.session.results)) {
    const fallbackName = getImageNameFromOutput(result);
    if (fallbackName) {
      return fallbackName;
    }
  }

  return null;
};
