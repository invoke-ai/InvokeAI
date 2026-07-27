import type { Project } from '@workbench/projectContracts';

/**
 * The project *document* codec: the wire and file shape of a project, and the
 * migrations that heal older ones. It deliberately knows nothing about the
 * Workbench reducer.
 *
 * Rehydrating a document into a live `Project` needs `normalizeWorkbenchProject`
 * from the aggregate state module, which transitively owns generation graphs,
 * widget state, and every policy the editor runs on. Keeping that step out of
 * this module is what lets the Launchpad read, write, and shape-check project
 * files without paying for the editor — see `deserializeProjectDocument` in
 * `./syncedPersistence` for the rehydrating half.
 */

/**
 * Undo/redo stacks are session-only (each entry is a full project snapshot,
 * far too heavy to autosave); everything else in the project document is the
 * project, verbatim.
 */
export const serializeProjectDocument = (project: Project): Record<string, unknown> => {
  const { undoRedo: _undoRedo, ...document } = project;

  return document;
};

const normalizeInvocationSourceId = (sourceId: unknown): unknown => {
  if (sourceId === 'project-graph') {
    return 'workflow';
  }

  if (sourceId === 'canvas-fill') {
    return 'canvas';
  }

  return sourceId;
};

export const normalizeLegacyProjectDocument = (data: Record<string, unknown>): Record<string, unknown> => {
  const invocation = data.invocation;
  const queue = data.queue;

  return {
    ...data,
    invocation:
      invocation && typeof invocation === 'object'
        ? { ...invocation, sourceId: normalizeInvocationSourceId((invocation as { sourceId?: unknown }).sourceId) }
        : invocation,
    queue:
      queue && typeof queue === 'object' && Array.isArray((queue as { items?: unknown }).items)
        ? {
            ...queue,
            items: (queue as { items: unknown[] }).items.map((item) => {
              if (!item || typeof item !== 'object') {
                return item;
              }

              const snapshot = (item as { snapshot?: unknown }).snapshot;

              return {
                ...item,
                snapshot:
                  snapshot && typeof snapshot === 'object'
                    ? {
                        ...snapshot,
                        sourceId: normalizeInvocationSourceId((snapshot as { sourceId?: unknown }).sourceId),
                      }
                    : snapshot,
              };
            }),
          }
        : queue,
  };
};

/**
 * The minimum a document must carry to be a project at all. A document that
 * fails this can never rehydrate, so callers that only need to reject junk
 * (an import picker, a file preview) can stop here instead of loading the
 * reducer to find out.
 */
export const isProjectDocumentShape = (data: Record<string, unknown>): boolean =>
  typeof data.id === 'string' &&
  typeof data.name === 'string' &&
  typeof data.layout === 'object' &&
  data.layout !== null;
