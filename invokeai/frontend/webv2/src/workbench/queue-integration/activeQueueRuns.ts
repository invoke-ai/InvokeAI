import type { Project } from '@workbench/projectContracts';

export const hasActiveQueueRuns = (project: Pick<Project, 'queue'>): boolean =>
  project.queue.items.some(
    (item) => item.status === 'pending' || item.status === 'running' || item.cancellationPending === true
  );

export const hasInFlightQueueRuns = (project: Pick<Project, 'queue'>): boolean =>
  project.queue.items.some(
    (item) =>
      item.status === 'running' ||
      item.cancellationPending === true ||
      (item.status === 'pending' && Boolean(item.backendBatchId || item.backendItemIds?.length))
  );
