import type { ProjectPushOutcome } from './projectFlush';

import { type OpenProjectHandle, registerOpenProject, unregisterOpenProject } from './syncStore';

/**
 * Publishes one {@link OpenProjectHandle} per open tab, for the life of the mounted editor.
 *
 * Derived from workbench state on every change rather than maintained by open and close call sites:
 * the registry decides whether the library mutates through the sync engine or over HTTP, so one
 * that disagreed with the tabs would put a write on the wrong side of that invariant.
 */
export interface OpenProjectBrokerDeps {
  /** Drop the tab. Called after the project is already gone from the server. */
  closeProject: (projectId: string) => void;
  /** Remove the project from the server, in the sync engine's own queue. */
  deleteProject: (projectId: string) => Promise<void>;
  /** Push this project's live document and report whether the server took it. */
  flushProject: (projectId: string) => Promise<ProjectPushOutcome>;
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
    markDeleted: () => deps.markProjectDeleted(projectId),
    // Renaming through the reducer first is what keeps the open document and the server's copy
    // telling the same story: the flush that follows carries the new name on the project's own
    // revision chain, where a library PUT would have landed beside it and forced a conflict fork.
    //
    // The flush outcome is deliberately ignored. The rename is already in the reducer and on the
    // local snapshot; a push that did not land is retried by the next save, and failing the rename
    // because the network blipped would undo nothing and explain less.
    rename: async (name: string) => {
      deps.renameProject(projectId, name);
      await deps.flushProject(projectId);
    },
    unmarkDeleted: () => deps.unmarkProjectDeleted(projectId),
  });

  /**
   * Publish the current open set, unregistering whatever is no longer in it.
   *
   * Registration is unconditional: the registry is cleared when the account changes, and a
   * `published` set that believed otherwise would never re-register, silently sending every library
   * mutation over HTTP for the rest of the mount. `published` records only what to *retract*.
   */
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
