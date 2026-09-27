/**
 * The project's workflow collection and its session edit histories. Documents are edited through structural
 * sharing so an inactive workflow keeps its identity while another is edited, and histories keep previous document
 * references rather than deep copies.
 */

import type {
  ProjectGraphState,
  ProjectWorkflowEntry,
  ProjectWorkflowRunPreview,
  ProjectWorkflowSource,
} from '@features/workflow/contracts';
import type { ProjectGraphAction } from '@features/workflow/utility';

import {
  createProjectGraph,
  createWorkflowId,
  getProjectGraphUndoEntry,
  projectGraphReducer,
  readLegacyPlaceholderGraph,
  readPersistedProjectGraph,
} from '@features/workflow/utility';

export interface ProjectWorkflowCollection {
  activeWorkflowId: string;
  entries: ProjectWorkflowEntry[];
}

export interface WorkflowHistoryEntry {
  id: string;
  createdAt: string;
  label: string;
  /** The document as it was before the edit; the reference the collection held, never a clone. */
  document: ProjectGraphState;
  /** Edits that arrive as a stream (typing, dragging) share a key so they fold into one step. */
  mergeKey?: string;
  /** When the entry last absorbed a same-key edit; the merge window runs from here. */
  mergedAt?: string;
  /** Global order across the project's histories; the aggregate cap evicts the lowest first. */
  sequence: number;
}

export interface WorkflowEditHistory {
  past: WorkflowHistoryEntry[];
  future: WorkflowHistoryEntry[];
}

export type ProjectWorkflowHistories = Record<string, WorkflowEditHistory>;

/** Session-only state the project carries beside its persisted collection. */
export interface ProjectWorkflowSession {
  workflows: ProjectWorkflowCollection;
  workflowHistories: ProjectWorkflowHistories;
}

/** One project keeps at most this many undo entries across all of its workflows. */
export const WORKFLOW_HISTORY_LIMIT = 40;
/** A pause this long between same-key edits (typing, dragging) starts a new undo step. */
export const WORKFLOW_UNDO_MERGE_WINDOW_MS = 1500;

const EMPTY_HISTORY: WorkflowEditHistory = { future: [], past: [] };

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export const createBlankWorkflowDocument = (): ProjectGraphState => createProjectGraph(createWorkflowId('workflow'));

export const createProjectWorkflowCollection = (document: ProjectGraphState): ProjectWorkflowCollection => ({
  activeWorkflowId: document.id,
  entries: [{ document }],
});

export const findProjectWorkflow = <Session extends ProjectWorkflowSession>(
  project: Session,
  workflowId: string
): ProjectWorkflowEntry | undefined => project.workflows.entries.find((entry) => entry.document.id === workflowId);

export const getActiveProjectWorkflow = <Session extends ProjectWorkflowSession>(
  project: Session
): ProjectWorkflowEntry => {
  const active = findProjectWorkflow(project, project.workflows.activeWorkflowId) ?? project.workflows.entries[0];

  if (!active) {
    throw new Error('A project always owns at least one workflow.');
  }

  return active;
};

/** The one active-workflow read: callers that need the editable graph go through here, never a second field. */
export const getActiveProjectGraph = <Session extends ProjectWorkflowSession>(project: Session): ProjectGraphState =>
  getActiveProjectWorkflow(project).document;

export const getProjectWorkflowHistory = <Session extends ProjectWorkflowSession>(
  project: Session,
  workflowId: string
): WorkflowEditHistory => project.workflowHistories[workflowId] ?? EMPTY_HISTORY;

// #region Persistence shapes

const normalizeSource = (candidate: unknown): ProjectWorkflowSource | null | undefined => {
  if (candidate === undefined) {
    return undefined;
  }

  if (!isRecord(candidate) || typeof candidate.libraryWorkflowId !== 'string' || !candidate.libraryWorkflowId) {
    return null;
  }

  const revision = candidate.revision;

  return {
    libraryWorkflowId: candidate.libraryWorkflowId,
    revision: Number.isSafeInteger(revision) && (revision as number) > 0 ? (revision as number) : null,
  };
};

/** A run preview is decoration: a damaged one is dropped rather than costing the project. */
const normalizeRunPreview = (candidate: unknown): ProjectWorkflowRunPreview | undefined => {
  if (
    !isRecord(candidate) ||
    typeof candidate.imageName !== 'string' ||
    !candidate.imageName ||
    typeof candidate.submittedAt !== 'string' ||
    typeof candidate.completedAt !== 'string'
  ) {
    return undefined;
  }

  return { completedAt: candidate.completedAt, imageName: candidate.imageName, submittedAt: candidate.submittedAt };
};

/**
 * Validates a persisted collection. Anything that would silently lose a workflow (a malformed entry, a repeated id,
 * an active id naming nothing) is refused rather than repaired.
 */
export const normalizeProjectWorkflowCollection = (candidate: unknown): ProjectWorkflowCollection | null => {
  if (!isRecord(candidate) || typeof candidate.activeWorkflowId !== 'string' || !Array.isArray(candidate.entries)) {
    return null;
  }

  if (candidate.entries.length === 0) {
    return null;
  }

  const entries: ProjectWorkflowEntry[] = [];
  const ids = new Set<string>();

  for (const raw of candidate.entries) {
    if (!isRecord(raw)) {
      return null;
    }

    // Authored content is refused when damaged, never replaced with a blank the user did not write.
    const document = readPersistedProjectGraph(raw.document);

    if (!document || ids.has(document.id)) {
      return null;
    }

    ids.add(document.id);

    const source = normalizeSource(raw.source);
    const lastRun = normalizeRunPreview(raw.lastRun);

    if (source === null) {
      return null;
    }

    entries.push({ document, ...(source ? { source } : {}), ...(lastRun ? { lastRun } : {}) });
  }

  if (!ids.has(candidate.activeWorkflowId)) {
    return null;
  }

  return { activeWorkflowId: candidate.activeWorkflowId, entries };
};

/**
 * A schema-2 document owned one graph. It becomes the first, active workflow; a library binding becomes source
 * metadata with an unknown revision, because nothing proves the old graph still matches the library's contents.
 */
/**
 * A legacy single graph becomes the first, active workflow. Only the Phase-1 placeholder (no authored content) is
 * converted into a blank; a damaged version-2 graph or any other unrecognised graph is refused (`null`).
 */
export const migrateProjectGraphToCollection = (projectGraph: unknown): ProjectWorkflowCollection | null => {
  const graph = isRecord(projectGraph) ? projectGraph : {};
  const { libraryWorkflowId, ...rest } = graph;
  const document = rest.version === 2 ? readPersistedProjectGraph(rest) : readLegacyPlaceholderGraph(rest);

  if (!document) {
    return null;
  }
  const source =
    typeof libraryWorkflowId === 'string' && libraryWorkflowId.length > 0
      ? { libraryWorkflowId, revision: null }
      : undefined;

  return { activeWorkflowId: document.id, entries: [{ document, ...(source ? { source } : {}) }] };
};

/** Portable documents carry no library write target: an archive or file never inherits one from embedded ids. */
export const stripProjectWorkflowSources = (collection: ProjectWorkflowCollection): ProjectWorkflowCollection =>
  collection.entries.some((entry) => entry.source)
    ? {
        ...collection,
        entries: collection.entries.map((entry) => {
          const { source: _source, ...rest } = entry;

          return rest;
        }),
      }
    : collection;

// #endregion

// #region Collection edits

const replaceEntry = <Session extends ProjectWorkflowSession>(
  project: Session,
  workflowId: string,
  update: (entry: ProjectWorkflowEntry) => ProjectWorkflowEntry
): Session => {
  const index = project.workflows.entries.findIndex((entry) => entry.document.id === workflowId);

  if (index === -1) {
    return project;
  }

  const entry = project.workflows.entries[index]!;
  const nextEntry = update(entry);

  if (nextEntry === entry) {
    return project;
  }

  const entries = project.workflows.entries.slice();

  entries[index] = nextEntry;

  return { ...project, workflows: { ...project.workflows, entries } };
};

/** A workflow nobody has authored into: the blank a fresh project starts with. Opening a template may take it over. */
export const isPlaceholderProjectWorkflow = (entry: ProjectWorkflowEntry): boolean => {
  const { document } = entry;
  const root = document.form.elements[document.form.rootElementId];

  return (
    entry.source === undefined &&
    entry.lastRun === undefined &&
    document.nodes.length === 0 &&
    document.edges.length === 0 &&
    document.description === '' &&
    document.notes === '' &&
    document.tags === '' &&
    document.author === '' &&
    document.contact === '' &&
    (document.name === '' || document.name === 'Untitled Workflow') &&
    Object.keys(document.form.elements).length === 1 &&
    root?.type === 'container' &&
    root.data.children.length === 0
  );
};

export interface AddProjectWorkflowOptions {
  source?: ProjectWorkflowSource;
  /** Take over the active workflow when it is an untouched placeholder instead of adding beside it. */
  reusePlaceholder?: boolean;
}

/** Adds and activates a workflow. The document id is the project copy's identity and must be new to the project. */
export const addProjectWorkflow = <Session extends ProjectWorkflowSession>(
  project: Session,
  document: ProjectGraphState,
  options: AddProjectWorkflowOptions = {}
): Session => {
  if (findProjectWorkflow(project, document.id)) {
    throw new Error(`The project already owns workflow ${document.id}.`);
  }

  const entry: ProjectWorkflowEntry = { document, ...(options.source ? { source: options.source } : {}) };
  const active = getActiveProjectWorkflow(project);
  const activeHistory = project.workflowHistories[active.document.id];
  // A blank that was edited and undone back to blank still has authored history worth keeping.
  const isUntouchedPlaceholder =
    isPlaceholderProjectWorkflow(active) &&
    (activeHistory === undefined || (activeHistory.past.length === 0 && activeHistory.future.length === 0));

  if (options.reusePlaceholder && isUntouchedPlaceholder) {
    const { [active.document.id]: _released, ...workflowHistories } = project.workflowHistories;

    return {
      ...project,
      workflowHistories,
      workflows: {
        activeWorkflowId: document.id,
        entries: project.workflows.entries.map((candidate) => (candidate === active ? entry : candidate)),
      },
    };
  }

  return {
    ...project,
    workflows: { activeWorkflowId: document.id, entries: [...project.workflows.entries, entry] },
  };
};

export const selectProjectWorkflow = <Session extends ProjectWorkflowSession>(
  project: Session,
  workflowId: string
): Session =>
  project.workflows.activeWorkflowId === workflowId || !findProjectWorkflow(project, workflowId)
    ? project
    : { ...project, workflows: { ...project.workflows, activeWorkflowId: workflowId } };

/** Copies a workflow under a fresh document id. Node ids may repeat: editor and execution state are scoped by workflow. */
export const duplicateProjectWorkflow = <Session extends ProjectWorkflowSession>(
  project: Session,
  workflowId: string,
  copyId: string,
  copyName: (name: string) => string
): Session => {
  const entry = findProjectWorkflow(project, workflowId);

  if (!entry) {
    return project;
  }

  const index = project.workflows.entries.indexOf(entry);
  const document: ProjectGraphState = structuredClone({
    ...entry.document,
    id: copyId,
    name: copyName(entry.document.name),
    updatedAt: new Date().toISOString(),
  });
  const copy: ProjectWorkflowEntry = { document, ...(entry.source ? { source: entry.source } : {}) };
  const entries = project.workflows.entries.slice();

  entries.splice(index + 1, 0, copy);

  return { ...project, workflows: { activeWorkflowId: copyId, entries } };
};

/**
 * Removes a workflow and releases its history. Selection moves to the next neighbour, or the previous one for the
 * last entry; removing the only workflow leaves a fresh blank in its place.
 */
export const removeProjectWorkflow = <Session extends ProjectWorkflowSession>(
  project: Session,
  workflowId: string
): Session => {
  const index = project.workflows.entries.findIndex((entry) => entry.document.id === workflowId);

  if (index === -1) {
    return project;
  }

  const { [workflowId]: _released, ...workflowHistories } = project.workflowHistories;
  const remaining = project.workflows.entries.filter((entry) => entry.document.id !== workflowId);

  if (remaining.length === 0) {
    const blank = createBlankWorkflowDocument();

    return { ...project, workflowHistories, workflows: createProjectWorkflowCollection(blank) };
  }

  const activeWorkflowId =
    project.workflows.activeWorkflowId === workflowId
      ? (remaining[Math.min(index, remaining.length - 1)]?.document.id ?? remaining[0]!.document.id)
      : project.workflows.activeWorkflowId;

  return { ...project, workflowHistories, workflows: { activeWorkflowId, entries: remaining } };
};

export const setProjectWorkflowSource = <Session extends ProjectWorkflowSession>(
  project: Session,
  workflowId: string,
  source: ProjectWorkflowSource | undefined
): Session =>
  replaceEntry(project, workflowId, (entry) => {
    if (entry.source?.libraryWorkflowId === source?.libraryWorkflowId && entry.source?.revision === source?.revision) {
      return entry;
    }

    const { source: _previous, ...rest } = entry;

    return source ? { ...rest, source } : rest;
  });

/** Records a successful run's output. An older submission that finishes later never replaces a newer one. */
export const recordProjectWorkflowRun = <Session extends ProjectWorkflowSession>(
  project: Session,
  workflowId: string,
  run: ProjectWorkflowRunPreview
): Session =>
  replaceEntry(project, workflowId, (entry) =>
    entry.lastRun && entry.lastRun.submittedAt > run.submittedAt ? entry : { ...entry, lastRun: run }
  );

// #endregion

// #region Document edits and history

let nextHistorySequence = 0;

/** Undo and redo entries share one budget; the entry created longest ago goes first, whichever side it sits on. */
const evictOldestAcrossHistories = (histories: ProjectWorkflowHistories): ProjectWorkflowHistories => {
  let total = 0;

  for (const history of Object.values(histories)) {
    total += history.past.length + history.future.length;
  }

  if (total <= WORKFLOW_HISTORY_LIMIT) {
    return histories;
  }

  let next = histories;

  while (total > WORKFLOW_HISTORY_LIMIT) {
    let oldestId: string | null = null;
    let oldestSide: 'future' | 'past' = 'past';
    let oldestSequence = Number.POSITIVE_INFINITY;

    for (const [workflowId, history] of Object.entries(next)) {
      const first = history.past[0];
      const deepest = history.future.at(-1);

      if (first && first.sequence < oldestSequence) {
        oldestSequence = first.sequence;
        oldestId = workflowId;
        oldestSide = 'past';
      }

      if (deepest && deepest.sequence < oldestSequence) {
        oldestSequence = deepest.sequence;
        oldestId = workflowId;
        oldestSide = 'future';
      }
    }

    if (oldestId === null) {
      break;
    }

    const history = next[oldestId]!;

    next = {
      ...next,
      [oldestId]:
        oldestSide === 'past'
          ? { ...history, past: history.past.slice(1) }
          : { ...history, future: history.future.slice(0, -1) },
    };
    total -= 1;
  }

  return next;
};

const pushHistory = <Session extends ProjectWorkflowSession>(
  project: Session,
  workflowId: string,
  previous: ProjectGraphState,
  label: string,
  mergeKey: string | undefined,
  timestamp: string
): Session => {
  const history = getProjectWorkflowHistory(project, workflowId);
  const last = history.past.at(-1);

  // An undo in between (`future` non-empty) ends the burst: the state the user just stood on must stay reachable as
  // its own step.
  if (
    mergeKey &&
    last?.mergeKey === mergeKey &&
    history.future.length === 0 &&
    Date.parse(timestamp) - Date.parse(last.mergedAt ?? last.createdAt) <= WORKFLOW_UNDO_MERGE_WINDOW_MS
  ) {
    return {
      ...project,
      workflowHistories: {
        ...project.workflowHistories,
        [workflowId]: { future: [], past: [...history.past.slice(0, -1), { ...last, mergedAt: timestamp }] },
      },
    };
  }

  nextHistorySequence += 1;

  const entry: WorkflowHistoryEntry = {
    createdAt: timestamp,
    document: previous,
    id: `undo-${nextHistorySequence.toString(36)}`,
    label,
    ...(mergeKey ? { mergeKey } : {}),
    sequence: nextHistorySequence,
  };

  return {
    ...project,
    workflowHistories: evictOldestAcrossHistories({
      ...project.workflowHistories,
      [workflowId]: { future: [], past: [...history.past, entry] },
    }),
  };
};

export interface ApplyWorkflowActionResult<Session> {
  project: Session;
  /** True when the reducer produced a new document. */
  didChange: boolean;
}

/** Applies a graph action to one workflow, recording an undo step when the action earns one. */
export const applyProjectWorkflowAction = <Session extends ProjectWorkflowSession>(
  project: Session,
  workflowId: string,
  action: ProjectGraphAction,
  timestamp = new Date().toISOString()
): ApplyWorkflowActionResult<Session> => {
  const entry = findProjectWorkflow(project, workflowId);

  if (!entry) {
    return { didChange: false, project };
  }

  const document = projectGraphReducer(entry.document, action);

  if (document === entry.document) {
    return { didChange: false, project };
  }

  const undoEntry = getProjectGraphUndoEntry(action);
  const withHistory = undoEntry
    ? pushHistory(project, workflowId, entry.document, undoEntry.label, undoEntry.mergeKey, timestamp)
    : project;

  return {
    didChange: true,
    project: replaceEntry(withHistory, workflowId, (current) => ({ ...current, document })),
  };
};

export const undoProjectWorkflow = <Session extends ProjectWorkflowSession>(
  project: Session,
  workflowId: string,
  timestamp = new Date().toISOString()
): Session => {
  const entry = findProjectWorkflow(project, workflowId);
  const history = getProjectWorkflowHistory(project, workflowId);
  const undone = history.past.at(-1);

  if (!entry || !undone) {
    return project;
  }

  nextHistorySequence += 1;

  const redoEntry: WorkflowHistoryEntry = {
    createdAt: timestamp,
    document: entry.document,
    id: `redo-${nextHistorySequence.toString(36)}`,
    label: undone.label,
    sequence: nextHistorySequence,
  };
  const restored = replaceEntry(project, workflowId, (current) => ({ ...current, document: undone.document }));

  return {
    ...restored,
    workflowHistories: {
      ...restored.workflowHistories,
      [workflowId]: {
        future: [redoEntry, ...history.future],
        past: history.past.slice(0, -1),
      },
    },
  };
};

export const redoProjectWorkflow = <Session extends ProjectWorkflowSession>(
  project: Session,
  workflowId: string,
  timestamp = new Date().toISOString()
): Session => {
  const entry = findProjectWorkflow(project, workflowId);
  const history = getProjectWorkflowHistory(project, workflowId);
  const redone = history.future[0];

  if (!entry || !redone) {
    return project;
  }

  nextHistorySequence += 1;

  const undoEntry: WorkflowHistoryEntry = {
    createdAt: timestamp,
    document: entry.document,
    id: `undo-${nextHistorySequence.toString(36)}`,
    label: redone.label,
    sequence: nextHistorySequence,
  };
  const restored = replaceEntry(project, workflowId, (current) => ({ ...current, document: redone.document }));

  return {
    ...restored,
    workflowHistories: evictOldestAcrossHistories({
      ...restored.workflowHistories,
      [workflowId]: { future: history.future.slice(1), past: [...history.past, undoEntry] },
    }),
  };
};

/** Replaces a document without touching history: seed advances and other non-undoable bookkeeping. */
export const setProjectWorkflowDocument = <Session extends ProjectWorkflowSession>(
  project: Session,
  workflowId: string,
  document: ProjectGraphState
): Session =>
  replaceEntry(project, workflowId, (entry) => (entry.document === document ? entry : { ...entry, document }));

// #endregion
