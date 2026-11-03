import type { ProjectEvent } from './projectEventContracts';

export const PROJECT_EVENT_LIMIT = 200;

export const prependProjectEvent = (events: readonly ProjectEvent[], event: ProjectEvent): ProjectEvent[] =>
  [event, ...events].slice(0, PROJECT_EVENT_LIMIT);
