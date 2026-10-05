import type { Project } from '@workbench/projectContracts';

import type { ProjectPushOutcome } from './projectFlush';

import { type OpenProjectHandle, registerOpenProject, unregisterOpenProject } from './syncStore';

/** Derive the broker from actual tabs so mutation routing cannot drift. */
export interface OpenProjectBrokerDeps {
  /** Drop the tab. Called after the project is already gone from the server. */
  closeProject: (projectId: string) => void;
  /** Remove the project from the server, in the sync engine's own queue. */
  deleteProject: (projectId: string) => Promise<void>;
  /** Push this project's live document and report whether the server took it. */
  flushProject: (projectId: string) => Promise<ProjectPushOutcome>;
  /** Persist the project's unsaved canvas pixels into its document. */
  flushPixels: (projectId: string) => Promise<void>;
  getProject: (projectId: string) => Project | undefined;
  getOpenProjectIds: () => string[];
  /** Stop the autosave recreating a project that is being deleted. */
  markProjectDeleted: (projectId: string) => void;
  /** Let it save again after a deletion that failed. */
  unmarkProjectDeleted: (projectId: string) => void;
  renameProject: (projectId: string, name: string) => void;
  subscribe: (listener: () => void) => () => void;
}

export interface OpenProjectBroker {
  dispose: () => void;
}

export const createOpenProjectBroker = (deps: OpenProjectBrokerDeps): OpenProjectBroker => {
  const published = new Set<string>();

  const buildHandle = (projectId: string): OpenProjectHandle => ({
    close: () => deps.closeProject(projectId),
    deleteOnServer: () => deps.deleteProject(projectId),
    flush: () => deps.flushProject(projectId),
    current: () => deps.getProject(projectId),
    flushPixels: () => deps.flushPixels(projectId),
    markDeleted: () => deps.markProjectDeleted(projectId),
    // Rename through the reducer's revision chain; an unacknowledged flush remains recoverable.
    rename: async (name: string) => {
      deps.renameProject(projectId, name);
      await deps.flushProject(projectId);
    },
    unmarkDeleted: () => deps.unmarkProjectDeleted(projectId),
  });

  /** Register unconditionally after account clears; published tracks only retraction. */
  const sync = (): void => {
    const openIds = new Set(deps.getOpenProjectIds());

    for (const projectId of published) {
      if (!openIds.has(projectId)) {
        unregisterOpenProject(projectId);
        published.delete(projectId);
      }
    }

    for (const projectId of openIds) {
      registerOpenProject(projectId, buildHandle(projectId));
      published.add(projectId);
    }
  };

  sync();

  const unsubscribe = deps.subscribe(sync);

  return {
    dispose: () => {
      unsubscribe();

      for (const projectId of published) {
        unregisterOpenProject(projectId);
      }

      published.clear();
    },
  };
};
