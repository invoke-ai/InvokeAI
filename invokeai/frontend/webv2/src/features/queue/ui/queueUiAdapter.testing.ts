import type { QueueUiAdapter } from './QueueUiContext';

const noLoadingModel = { getLabel: (): string | null => null, subscribe: () => () => undefined };

/** A single-user adapter that can manage everything, connected, with nothing queued locally. */
export const createTestQueueUiAdapter = (overrides: Partial<QueueUiAdapter> = {}): QueueUiAdapter => ({
  ItemActions: () => null,
  activeProjectId: null,
  activeProjectName: null,
  canManageInvocationCache: false,
  canManageItem: () => true,
  canManageProcessor: true,
  canViewItemDetails: () => true,
  cancelLocalUnsubmittedRuns: () => undefined,
  isConnected: true,
  isProjectOpen: () => true,
  loadingModel: noLoadingModel,
  localUnsubmittedRuns: { activeProject: 0, allProjects: 0 },
  notify: { error: () => undefined, info: () => undefined, success: () => undefined },
  openQueue: () => undefined,
  preloadItemActions: () => undefined,
  queueJobsScope: 'all',
  viewer: 'single',
  ...overrides,
});
