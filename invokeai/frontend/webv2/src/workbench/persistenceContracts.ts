import type { RefusedWorkbenchProject, WorkbenchState } from './projectContracts';

export interface HydratedWorkbenchSnapshot {
  version: 1;
  savedAt: string;
  state: WorkbenchState;
  /** Persisted projects this client refused to load. They are absent from `state`. */
  refusedProjects: RefusedWorkbenchProject[];
}
