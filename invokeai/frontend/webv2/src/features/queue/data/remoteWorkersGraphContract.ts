import type { QueueBackendGraph } from '@features/queue/core/types';

export const AUTOMATIC_REMOTE_WORKER_NODE_ID = '__irw_remote_worker_dispatch__';
export const AUTOMATIC_REMOTE_WORKER_NODE_TYPE = 'irw_builtin_remote_worker_dispatch';

export const hasRemoteWorkerDispatchNode = (graph: QueueBackendGraph): boolean =>
  Object.values(graph.nodes).some((node) => node.type === AUTOMATIC_REMOTE_WORKER_NODE_TYPE);
