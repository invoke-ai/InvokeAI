import type { Project } from '@workbench/projectContracts';
import type { ProjectWorkflowCollection } from '@workbench/projectWorkflows';
import type { LayeredSettingKey } from '@workbench/settings/projectSettings';

import { stripInfiniteWindowAnchor, stripSessionScopedGallerySearch } from '@features/gallery/contracts';
import { migrateProjectGraphToCollection, normalizeProjectWorkflowCollection } from '@workbench/projectWorkflows';
import { getLayeredSettings } from '@workbench/settings/projectSettings';

import { PROJECT_DOCUMENT_SCHEMA_VERSION } from './documentSchemaVersion';

/** Keep document codecs reducer-free; load rehydration lazily. */

export const PROJECT_DOCUMENT_MAX_BYTES = 32 * 1024 * 1024;

export type ProjectDocumentV3 = Omit<
  Pick<
    Project,
    | 'canvas'
    | 'floatingWidgets'
    | 'id'
    | 'invocation'
    | 'lastFloatingGeometry'
    | 'layout'
    | 'name'
    | 'presetWorkingLayouts'
    | 'promptHistory'
    | 'settings'
    | 'widgetGraphs'
    | 'widgetInstances'
    | 'widgetRegions'
  >,
  'floatingWidgets' | 'lastFloatingGeometry' | 'presetWorkingLayouts'
> & {
  documentSchemaVersion: typeof PROJECT_DOCUMENT_SCHEMA_VERSION;
  floatingWidgets?: Project['floatingWidgets'];
  lastFloatingGeometry?: Project['lastFloatingGeometry'];
  /** Optional and additive, so schema 3 holds: a build that predates it ignores it and leaves it out of its next save. */
  presetWorkingLayouts?: Project['presetWorkingLayouts'];
  /**
   * Present once the document's layerable settings were coerced to pins (see `migrateProjectDocument`); every save
   * of this build writes `SETTINGS_LAYERING_VERSION`. Additive like `presetWorkingLayouts`. A build predating the
   * marker drops it on save. The next load then coerces again: a pin equal to the built-in is dropped, and the
   * project follows the user's default for that setting. Accepted: only a project edited in such a build, and only
   * pins equal to the built-in. A higher marker from a newer build counts as layered here, but this build's saves
   * write its own version, so the newer build re-runs its steps above that version.
   */
  settingsLayering?: number;
  workflows: ProjectWorkflowCollection;
};

/**
 * The `settingsLayering` marker's value. This build accepts a higher marker but saves this one, so a newer build
 * re-runs every step above it on a document it already migrated: each future step must be idempotent (re-running it
 * changes nothing). Another drop of values equal to the built-in, say, would damage pins set in the newer build.
 */
export const SETTINGS_LAYERING_VERSION = 1;

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
    ...(persistent.lastFloatingGeometry ? { lastFloatingGeometry: persistent.lastFloatingGeometry } : {}),
    layout: persistent.layout,
    name: persistent.name,
    ...(persistent.presetWorkingLayouts ? { presetWorkingLayouts: persistent.presetWorkingLayouts } : {}),
    promptHistory: persistent.promptHistory,
    settings: persistent.settings,
    settingsLayering: SETTINGS_LAYERING_VERSION,
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

  return {
    ...document,
    documentSchemaVersion: PROJECT_DOCUMENT_SCHEMA_VERSION,
    settingsLayering: SETTINGS_LAYERING_VERSION,
  };
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

/**
 * Drops `values` keys the owner read as the built-in when settings first layered (`getLayeredSettings`, frozen at that
 * version); null when none does.
 */
const withoutBuiltInValues = (
  values: Record<string, unknown>,
  layered: Readonly<Record<string, LayeredSettingKey>>
): Record<string, unknown> | null => {
  const redundant = Object.keys(layered).filter(
    (key) => Object.hasOwn(values, key) && layered[key]!.read(values[key]) === layered[key]!.builtIn
  );
  if (redundant.length === 0) {
    return null;
  }
  const next = { ...values };
  for (const key of redundant) {
    delete next[key];
  }
  return next;
};

const coerceWidgetInstances = (instances: Record<string, unknown>): Record<string, unknown> | null => {
  const builtIns = getLayeredSettings().widgetValues;
  let didChange = false;
  const next: Record<string, unknown> = {};
  for (const [instanceId, instance] of Object.entries(instances)) {
    const owner = isRecord(instance) && typeof instance.typeId === 'string' ? builtIns[instance.typeId] : undefined;
    const state = isRecord(instance) ? instance.state : undefined;
    const values =
      owner && isRecord(state) && isRecord(state.values) ? withoutBuiltInValues(state.values, owner) : null;
    if (values && isRecord(state)) {
      didChange = true;
      next[instanceId] = { ...(instance as Record<string, unknown>), state: { ...state, values } };
    } else {
      next[instanceId] = instance;
    }
  }
  return didChange ? next : null;
};

const coerceProjectSettings = (settings: Record<string, unknown>): Record<string, unknown> => {
  const { antialiasProgressImages: _antialias, ...rest } = settings;
  return withoutBuiltInValues(rest, getLayeredSettings().projectSettings) ?? rest;
};

/**
 * Before settings layered over the user's defaults, a stored value equal to the built-in read the same as an absent
 * one, so a document without the marker drops those (widget values its owner's settings edit, and the generation
 * flags) and the retired per-project antialias flag: what is left is what the project chose. With the marker (this
 * version, or a newer build's higher one) the document's values are its pins and stay as they are.
 */
const coerceSettingsLayering = (document: Record<string, unknown>): Record<string, unknown> => {
  if (typeof document.settingsLayering === 'number' && document.settingsLayering >= SETTINGS_LAYERING_VERSION) {
    return document;
  }
  const widgetInstances = isRecord(document.widgetInstances) ? coerceWidgetInstances(document.widgetInstances) : null;
  return {
    ...document,
    ...(isRecord(document.settings) ? { settings: coerceProjectSettings(document.settings) } : {}),
    settingsLayering: SETTINGS_LAYERING_VERSION,
    ...(widgetInstances ? { widgetInstances } : {}),
  };
};

export type ProjectDocumentMigration =
  | { status: 'migrated'; document: Record<string, unknown> }
  | { status: 'malformed'; reason: string };

/**
 * The migration boundary: any supported older document comes out as schema 3. A document without a version is
 * schema 1; schema 2 owned one `projectGraph`, which becomes the first, active workflow. Collections that would
 * lose a workflow if repaired are refused instead. A document without `settingsLayering` has its stored built-in
 * settings values dropped once (see `coerceSettingsLayering`). Callers check for a future version before calling this.
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
      ...coerceSettingsLayering(rest),
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
