import type { Project } from '@workbench/projectContracts';
import type { ProjectWorkflowCollection } from '@workbench/projectWorkflows';

import { stripInfiniteWindowAnchor, stripSessionScopedGallerySearch } from '@features/gallery/contracts';
import { migrateProjectGraphToCollection, normalizeProjectWorkflowCollection } from '@workbench/projectWorkflows';

/** Keep document codecs reducer-free; load rehydration lazily. */

export const PROJECT_DOCUMENT_SCHEMA_VERSION = 3;
export const PROJECT_DOCUMENT_MAX_BYTES = 32 * 1024 * 1024;

export type ProjectDocumentV3 = Omit<
  Pick<
    Project,
    | 'canvas'
    | 'floatingWidgets'
    | 'id'
    | 'invocation'
    | 'layout'
    | 'name'
    | 'promptHistory'
    | 'settings'
    | 'widgetGraphs'
    | 'widgetInstances'
    | 'widgetRegions'
  >,
  'floatingWidgets'
> & {
  documentSchemaVersion: typeof PROJECT_DOCUMENT_SCHEMA_VERSION;
  floatingWidgets?: Project['floatingWidgets'];
  workflows: ProjectWorkflowCollection;
};

export const stripSessionScopedGalleryState = (project: Project): Project => {
  let didChange = false;
  const widgetInstances = Object.fromEntries(
    Object.entries(project.widgetInstances).map(([instanceId, instance]) => {
      if (instance.typeId !== 'gallery') {
        return [instanceId, instance];
      }
      const values = instance.state.values;
      const strippedValues = stripSessionScopedGallerySearch(values);
      const strippedAnchorValues = stripInfiniteWindowAnchor(strippedValues ?? values);
      if (strippedValues === null && strippedAnchorValues === null) {
        return [instanceId, instance];
      }
      didChange = true;
      return [
        instanceId,
        { ...instance, state: { ...instance.state, values: strippedAnchorValues ?? strippedValues ?? values } },
      ];
    })
  );

  return didChange ? { ...project, widgetInstances } : project;
};

/**
 * The durable shape server persistence writes: an allowlist of editable fields. Queue runs, events, undo and workflow
 * histories are session state and never appear.
 */
export const serializeProjectDocumentV3 = (project: Project): ProjectDocumentV3 => {
  const persistent = stripSessionScopedGalleryState(project);
  const document: ProjectDocumentV3 = {
    canvas: persistent.canvas,
    documentSchemaVersion: PROJECT_DOCUMENT_SCHEMA_VERSION,
    ...(persistent.floatingWidgets ? { floatingWidgets: persistent.floatingWidgets } : {}),
    id: persistent.id,
    invocation: persistent.invocation,
    layout: persistent.layout,
    name: persistent.name,
    promptHistory: persistent.promptHistory,
    settings: persistent.settings,
    widgetGraphs: persistent.widgetGraphs,
    widgetInstances: persistent.widgetInstances,
    widgetRegions: persistent.widgetRegions,
    workflows: persistent.workflows,
  };

  return document;
};

export const serializeProjectDocumentV3Json = (
  project: Project
): { byteSize: number; document: ProjectDocumentV3; documentJson: string } => {
  const document = serializeProjectDocumentV3(project);
  const documentJson = JSON.stringify(document);

  return { byteSize: new TextEncoder().encode(documentJson).byteLength, document, documentJson };
};

/**
 * The transfer shape export, import and duplication write: every field a loaded project carries except session
 * state, so a document authored with fields this client does not know keeps them, stamped with the current schema.
 */
export const serializeProjectDocument = (project: Project): Record<string, unknown> => {
  const {
    events: _events,
    graphHistory: _graphHistory,
    queue: _queue,
    undoRedo: _undoRedo,
    workflowHistories: _workflowHistories,
    ...document
  } = stripSessionScopedGalleryState(project) as Project & { graphHistory?: unknown };

  return { ...document, documentSchemaVersion: PROJECT_DOCUMENT_SCHEMA_VERSION };
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

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export type ProjectDocumentMigration =
  | { status: 'migrated'; document: Record<string, unknown> }
  | { status: 'malformed'; reason: string };

/**
 * The migration boundary: any supported older document comes out as schema 3. A document without a version is
 * schema 1; schema 2 owned one `projectGraph`, which becomes the first, active workflow. Collections that would
 * lose a workflow if repaired are refused instead. Callers check for a future version before calling this.
 */
export const migrateProjectDocument = (data: Record<string, unknown>): ProjectDocumentMigration => {
  const version = data.documentSchemaVersion;

  if (version !== undefined && version !== 2 && version !== PROJECT_DOCUMENT_SCHEMA_VERSION) {
    return { reason: `Unsupported project document schema ${String(version)}.`, status: 'malformed' };
  }

  if (typeof data.id !== 'string' || typeof data.name !== 'string' || !isRecord(data.layout)) {
    return { reason: 'The document is missing its project identity or layout.', status: 'malformed' };
  }

  const {
    events: _events,
    graphHistory: _graphHistory,
    invocation,
    projectGraph,
    queue: _queue,
    undoRedo: _undoRedo,
    workflowHistories: _workflowHistories,
    workflows,
    ...rest
  } = data;

  let collection: ProjectWorkflowCollection;

  if (version === PROJECT_DOCUMENT_SCHEMA_VERSION) {
    const normalized = normalizeProjectWorkflowCollection(workflows);

    if (!normalized) {
      return {
        reason: 'The workflow collection is malformed or names a missing active workflow.',
        status: 'malformed',
      };
    }

    collection = normalized;
  } else {
    const migrated = migrateProjectGraphToCollection(projectGraph);

    if (!migrated) {
      return { reason: 'The project graph is damaged.', status: 'malformed' };
    }

    collection = migrated;
  }

  return {
    document: {
      ...rest,
      documentSchemaVersion: PROJECT_DOCUMENT_SCHEMA_VERSION,
      invocation: isRecord(invocation)
        ? { ...invocation, sourceId: normalizeInvocationSourceId(invocation.sourceId) }
        : invocation,
      workflows: collection,
    },
    status: 'migrated',
  };
};

const patchGalleryValues = (
  values: Record<string, unknown>,
  boardId: string,
  selectBoard: boolean
): Record<string, unknown> => ({
  ...values,
  projectBoardId: boardId,
  ...(selectBoard ? { selectedBoardId: boardId } : {}),
});

/**
 * Cache the server-authoritative board ID in either persisted widget shape. selectBoard selects it on first open;
 * rehydration preserves the user's destination. Leave documents without gallery state untouched.
 */
export const applyAuthoritativeProjectBoard = (
  projectDocument: Record<string, unknown>,
  boardId: string,
  options: { selectBoard: boolean }
): Record<string, unknown> => {
  let hasChanged = false;
  const next: Record<string, unknown> = { ...projectDocument };

  const instances = projectDocument.widgetInstances;
  if (isRecord(instances)) {
    const nextInstances: Record<string, unknown> = {};

    for (const [instanceId, instance] of Object.entries(instances)) {
      const state = isRecord(instance) ? instance.state : null;

      if (!isRecord(instance) || instance.typeId !== 'gallery' || !isRecord(state)) {
        nextInstances[instanceId] = instance;
        continue;
      }

      const values = isRecord(state.values) ? state.values : {};

      nextInstances[instanceId] = {
        ...instance,
        state: { ...state, values: patchGalleryValues(values, boardId, options.selectBoard) },
      };
      hasChanged = true;
    }

    next.widgetInstances = nextInstances;
  }

  const states = projectDocument.widgetStates;
  if (isRecord(states) && isRecord(states.gallery)) {
    const gallery = states.gallery;
    const values = isRecord(gallery.values) ? gallery.values : {};

    next.widgetStates = {
      ...states,
      gallery: { ...gallery, values: patchGalleryValues(values, boardId, options.selectBoard) },
    };
    hasChanged = true;
  }

  return hasChanged ? next : projectDocument;
};
