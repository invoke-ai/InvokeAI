import type { QueueLocalRun, QueueUiAdapter } from './QueueUiContext';

const noLoadingModel = { getLabel: (): string | null => null, subscribe: () => () => undefined };

/** A single-user adapter that can manage everything, connected, with nothing queued locally. */
export const createTestQueueUiAdapter = (overrides: Partial<QueueUiAdapter> = {}): QueueUiAdapter => ({
  ItemActions: () => null,
  activeProjectId: null,
  activeProjectName: null,
  canManageInvocationCache: false,
  canManageModelCache: false,
  canManageItem: () => true,
  canManageProcessor: true,
  canViewItemDetails: () => true,
  cancelLocalRun: () => undefined,
  cancelLocalRuns: () => undefined,
  isConnected: true,
  isProjectOpen: () => true,
  loadingModel: noLoadingModel,
  localRuns: [],
  notify: { error: () => undefined, info: () => undefined, success: () => undefined },
  openQueue: () => undefined,
  preloadItemActions: () => undefined,
  queueJobsScope: 'all',
  viewer: 'single',
  ...overrides,
});

/** A run this tab queued long ago in `project-1` and is still submitting, with no failure yet. */
export const createTestLocalRun = (overrides: Partial<QueueLocalRun> = {}): QueueLocalRun => ({
  id: 'run-1',
  label: 'A castle at dusk',
  projectId: 'project-1',
  retry: null,
  submittedAt: 0,
  ...overrides,
});
