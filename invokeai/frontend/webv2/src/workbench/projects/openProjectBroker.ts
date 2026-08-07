import { type OpenProjectHandle, registerOpenProject, unregisterOpenProject } from './syncStore';

/**
 * Publishes one {@link OpenProjectHandle} per open tab, for the whole life of the mounted editor.
 *
 * The registry has to say exactly which projects the workbench holds, because that is what decides
 * whether the library mutates a project through the sync engine or over HTTP — and a registry that
 * disagreed with the tabs would put a write on the wrong side of the invariant. So it is derived
 * from workbench state on every change rather than maintained by open and close call sites, which
 * is a rule that cannot drift as new ways to open a project are added.
 *
 * The dependencies are plain functions rather than the store and the persistence service. This
 * module belongs to the project layer, which the workbench shell wires up; taking the aggregates
 * themselves would invert that.
 */
export interface OpenProjectBrokerDeps {
  /** Drop the tab. Called after the project is already gone from the server. */
  closeProject: (projectId: string) => void;
  /** Push this project's live document and wait for the acknowledgement. */
  flushProject: (projectId: string) => Promise<void>;
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
    flush: () => deps.flushProject(projectId),
    markDeleted: () => deps.markProjectDeleted(projectId),
    // Renaming through the reducer first is what keeps the open document and the server's copy
    // telling the same story: the flush that follows carries the new name on the project's own
    // revision chain, where a library PUT would have landed beside it and forced a conflict fork.
    rename: async (name: string) => {
      deps.renameProject(projectId, name);
      await deps.flushProject(projectId);
    },
    unmarkDeleted: () => deps.unmarkProjectDeleted(projectId),
  });

  const sync = (): void => {
    const openIds = new Set(deps.getOpenProjectIds());

    for (const projectId of published) {
      if (!openIds.has(projectId)) {
        unregisterOpenProject(projectId);
        published.delete(projectId);
      }
    }

    for (const projectId of openIds) {
      if (!published.has(projectId)) {
        registerOpenProject(projectId, buildHandle(projectId));
        published.add(projectId);
      }
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
