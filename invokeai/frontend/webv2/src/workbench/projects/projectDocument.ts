import type { Project } from '@workbench/projectContracts';
import type { ProjectWorkflowCollection } from '@workbench/projectWorkflows';
import type { LayeredSettingKey } from '@workbench/settings/projectSettings';
import type { StoredProjectLayout } from '@workbench/workspace/migrateProjectLayout';

import { stripInfiniteWindowAnchor, stripSessionScopedGallerySearch } from '@features/gallery/contracts';
import { migrateProjectGraphToCollection, normalizeProjectWorkflowCollection } from '@workbench/projectWorkflows';
import { getLayeredSettings } from '@workbench/settings/projectSettings';

import { PROJECT_DOCUMENT_SCHEMA_VERSION } from './documentSchemaVersion';

/** Keep document codecs reducer-free; load rehydration lazily. */

export const PROJECT_DOCUMENT_MAX_BYTES = 32 * 1024 * 1024;

export type ProjectDocumentV3 = Pick<
  Project,
  'canvas' | 'id' | 'invocation' | 'name' | 'promptHistory' | 'settings' | 'widgetGraphs' | 'widgetInstances'
> & {
  documentSchemaVersion: typeof PROJECT_DOCUMENT_SCHEMA_VERSION;
  lastFloatingGeometry?: Project['lastFloatingGeometry'];
  /** Additive; an older build drops it, and the next load coerces again (accepted: pins equal to the built-in). */
  settingsLayering?: number;
  workflows: ProjectWorkflowCollection;
  /** The workspace the project is on, and how it looks at each; a document without them stored the layout below. */
  workspaceId?: Project['workspaceId'];
  views?: Project['views'];
  /** Additive: workspaces whose own arrangement the project lost when its stored layout migrated, until it is told. */
  droppedArrangements?: Project['droppedArrangements'];
} & Partial<StoredLayoutFields>;

/**
 * The layout a document stored before structure and view split. A project loaded from one writes it back unchanged
 * until the editor migrates it against the account, so a copy, import or export made meanwhile keeps it. `heldView` is
 * additive within it: what a held project shows, which a build before the split drops on save.
 */
interface StoredLayoutFields {
  floatingWidgets: StoredProjectLayout['floatingWidgets'];
  heldView: StoredProjectLayout['heldView'];
  layout: StoredProjectLayout['layout'];
  presetWorkingLayouts: StoredProjectLayout['presetWorkingLayouts'];
  widgetRegions: StoredProjectLayout['widgetRegions'];
}

/**
 * The layout fields a project's document holds: its workspace and views, or the stored layout it has not migrated or
 * holds until its account's adoption is saved (`heldLegacyLayout`), which a later load migrates the same way again and
 * then shows as it was held.
 */
const toDocumentLayoutFields = ({
  droppedArrangements,
  heldLegacyLayout,
  legacyLayout: unmigrated,
  views,
  workspaceId,
}: Pick<Project, 'droppedArrangements' | 'heldLegacyLayout' | 'legacyLayout' | 'views' | 'workspaceId'>): Pick<
  ProjectDocumentV3,
  | 'droppedArrangements'
  | 'floatingWidgets'
  | 'heldView'
  | 'layout'
  | 'presetWorkingLayouts'
  | 'views'
  | 'widgetRegions'
  | 'workspaceId'
> => {
  const legacyLayout = unmigrated ?? heldLegacyLayout;
  const heldView: StoredProjectLayout['heldView'] = unmigrated
    ? unmigrated.heldView
    : { ...(droppedArrangements ? { droppedArrangements: [...droppedArrangements] } : {}), views, workspaceId };

  return legacyLayout
    ? {
        ...(legacyLayout.floatingWidgets ? { floatingWidgets: legacyLayout.floatingWidgets } : {}),
        ...(heldView ? { heldView } : {}),
        layout: legacyLayout.layout,
        ...(legacyLayout.presetWorkingLayouts ? { presetWorkingLayouts: legacyLayout.presetWorkingLayouts } : {}),
        widgetRegions: legacyLayout.widgetRegions,
      }
    : { ...(droppedArrangements ? { droppedArrangements: [...droppedArrangements] } : {}), views, workspaceId };
};

/** Newer builds re-run steps above this on documents they already migrated: every step must be idempotent. */
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

/** An allowlist of editable fields; queue runs, events, undo and workflow histories never appear. */
export const serializeProjectDocumentV3 = (project: Project): ProjectDocumentV3 => {
  const persistent = stripSessionScopedGalleryState(project);
  const document: ProjectDocumentV3 = {
    canvas: persistent.canvas,
    documentSchemaVersion: PROJECT_DOCUMENT_SCHEMA_VERSION,
    id: persistent.id,
    invocation: persistent.invocation,
    ...(persistent.lastFloatingGeometry ? { lastFloatingGeometry: persistent.lastFloatingGeometry } : {}),
    ...toDocumentLayoutFields(persistent),
    name: persistent.name,
    promptHistory: persistent.promptHistory,
    settings: persistent.settings,
    settingsLayering: SETTINGS_LAYERING_VERSION,
    widgetGraphs: persistent.widgetGraphs,
    widgetInstances: persistent.widgetInstances,
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

/** Keeps fields this client does not know, minus session state, stamped with the current schema. */
export const serializeProjectDocument = (project: Project): Record<string, unknown> => {
  const {
    droppedArrangements,
    events: _events,
    graphHistory: _graphHistory,
    heldLegacyLayout,
    legacyLayout,
    queue: _queue,
    undoRedo: _undoRedo,
    views,
    workflowHistories: _workflowHistories,
    workspaceId,
    ...document
  } = stripSessionScopedGalleryState(project) as Project & { graphHistory?: unknown };

  return {
    ...document,
    ...toDocumentLayoutFields({ droppedArrangements, heldLegacyLayout, legacyLayout, views, workspaceId }),
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

/** Without the marker, stored values equal to the built-in read as absent before layering, so they are dropped. */
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

/** Any supported older document comes out as schema 3; callers check for a future version first. */
export const migrateProjectDocument = (data: Record<string, unknown>): ProjectDocumentMigration => {
  const version = data.documentSchemaVersion;

  if (version !== undefined && version !== 2 && version !== PROJECT_DOCUMENT_SCHEMA_VERSION) {
    return { reason: `Unsupported project document schema ${String(version)}.`, status: 'malformed' };
  }

  if (
    typeof data.id !== 'string' ||
    typeof data.name !== 'string' ||
    (typeof data.workspaceId !== 'string' && !isRecord(data.layout))
  ) {
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

/** Cache the server-authoritative board ID; rehydration preserves the user's destination. */
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
