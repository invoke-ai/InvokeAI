import type { QueueGenerationMeta, QueueReadModel } from '@features/queue/contracts';
import type { CanvasCandidateSlot } from '@workbench/canvasStagingView';

import {
  buildProjectQueueItemOriginPrefix,
  buildQueueItemOrigin,
  extractGenerationMeta,
} from '@features/queue/contracts';
import { getQueueReadModelOptions } from '@features/queue/queries';
import { useQuery } from '@tanstack/react-query';
import { QueueRecallMenuGroup } from '@workbench/queue-integration/QueueRecallMenuGroup';
import { useQueueItemRecall } from '@workbench/queue-integration/useQueueItemRecall';
import { useActiveProjectId } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';

/** A staged result carries no executed field values; the local submission snapshot is the source. */
const NO_META: QueueGenerationMeta = {};

/** Read executed prompts and seed from queue metadata; randomized seeds are unavailable from authored settings. */
const useStagedCandidateMeta = (projectId: string, backendItemId: number | undefined): QueueGenerationMeta => {
  const scope = useMemo(() => ({ originPrefix: buildProjectQueueItemOriginPrefix(projectId) }), [projectId]);
  const select = useCallback(
    (model: QueueReadModel): QueueGenerationMeta => {
      const item = model.items.find((candidate) => candidate.id === backendItemId);
      return item ? extractGenerationMeta(item) : NO_META;
    },
    [backendItemId]
  );
  return useQuery({ ...getQueueReadModelOptions(scope), enabled: backendItemId !== undefined, select }).data ?? NO_META;
};

/** The staged result's recall verbs, read from the queue item that produced it. */
export const StagingItemRecallGroup = ({ slot }: { slot: CanvasCandidateSlot }) => {
  const projectId = useActiveProjectId();
  const meta = useStagedCandidateMeta(projectId, slot.candidate.sourceBackendItemId);
  const recall = useQueueItemRecall(buildQueueItemOrigin(slot.queueItemId, projectId), meta);

  return <QueueRecallMenuGroup recall={recall} />;
};
