import type { QueueBackendGraph, QueueResultDestination } from '@features/queue/core/types';

import {
  AUTOMATIC_REMOTE_WORKER_NODE_ID,
  AUTOMATIC_REMOTE_WORKER_NODE_TYPE,
  hasRemoteWorkerDispatchNode,
} from './remoteWorkersGraphContract';
import {
  getRemoteWorkerName,
  getRemoteWorkerUrls,
  getRemoteWorkersSettings,
  isRemoteWorkerEnabled,
} from './remoteWorkersStore';

/** Only the submitted graph is modified. The saved Canvas/workflow stays untouched. */
export const applyRemoteWorkersToGraph = (
  graph: QueueBackendGraph,
  galleryBoardId: string | null,
  destination: QueueResultDestination
): QueueBackendGraph => {
  const settings = getRemoteWorkersSettings();
  if (!settings.enabled || (destination !== 'gallery' && destination !== 'canvas')) {
    return graph;
  }

  const configuredUrls = getRemoteWorkerUrls(settings.workerUrls);
  // Availability is backend-owned; browser health is display-only.
  const urls = configuredUrls.filter(isRemoteWorkerEnabled);
  if ((urls.length === 0 && settings.dispatchMode !== 'remote_only') || hasRemoteWorkerDispatchNode(graph)) {
    return graph;
  }
  if (Object.hasOwn(graph.nodes, AUTOMATIC_REMOTE_WORKER_NODE_ID)) {
    return graph;
  }

  const names = urls.map((url) => getRemoteWorkerName(url, configuredUrls.indexOf(url)));
  const [remoteUrl, ...additionalUrls] = urls;

  const kickoff = {
    id: AUTOMATIC_REMOTE_WORKER_NODE_ID,
    type: AUTOMATIC_REMOTE_WORKER_NODE_TYPE,
    use_cache: false,
    is_intermediate: true,
    remote_url: remoteUrl,
    additional_remote_urls: additionalUrls.join('\n'),
    remote_worker_names: JSON.stringify(names),
    dispatch_mode: settings.dispatchMode === 'remote_only' ? 'Remote Only' : 'Distributed',
    auto_transfer_missing_models: settings.autoTransferMissingModels,
    model_transfer_host: settings.modelTransferHost.trim(),
    keep_remote_copies: settings.keepRemoteCopies,
    result_destination: destination,
    local_gallery_board_id:
      destination === 'gallery' && galleryBoardId && galleryBoardId !== 'none' ? galleryBoardId : '',
  };

  // Both modes keep the real executable graph in the local queue row. Distributed
  // lets Local race the backend worker pool for it. Remote Only is parked by the
  // backend; if Local wins the enqueue race, the AAA helper hands it to a remote
  // and skips the remaining local nodes after the remote completes.
  return {
    ...graph,
    nodes: { ...graph.nodes, [AUTOMATIC_REMOTE_WORKER_NODE_ID]: kickoff },
  };
};
