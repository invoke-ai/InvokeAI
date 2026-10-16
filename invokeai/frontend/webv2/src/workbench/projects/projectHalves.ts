import type { ProjectDocumentMigration, ProjectDocumentV3 } from './projectDocument';

import { PROJECT_DOCUMENT_SCHEMA_VERSION, PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION } from './documentSchemaVersion';

/*
 * The project document as the server stores it from schema 4: the project **content** (what the project is, shared
 * with every member) and each member's **workspace** (how they work on it). The browser keeps one whole document per
 * project; the split happens only here, at the server boundary.
 */

type CanvasState = ProjectDocumentV3['canvas'];
type ProjectWorkflowEntry = ProjectDocumentV3['workflows']['entries'][number];
type ProjectWorkflowRunPreview = NonNullable<ProjectWorkflowEntry['lastRun']>;

export interface ProjectContentV4 {
  documentSchemaVersion: typeof PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION;
  id: string;
  name: string;
  /** Without the staging area, and with no layer selected: both are the workspace's. */
  canvas: Omit<CanvasState, 'stagingArea'>;
  /** Each entry without its last run, which belongs to whoever ran it. */
  workflows: { entries: Omit<ProjectWorkflowEntry, 'lastRun'>[] };
}

export interface ProjectWorkspaceV4 extends Pick<
  ProjectDocumentV3,
  | 'floatingWidgets'
  | 'invocation'
  | 'lastFloatingGeometry'
  | 'layout'
  | 'presetWorkingLayouts'
  | 'promptHistory'
  | 'settings'
  | 'widgetGraphs'
  | 'widgetInstances'
  | 'widgetRegions'
> {
  documentSchemaVersion: typeof PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION;
  canvas: { selectedLayerId: string | null; stagingArea: CanvasState['stagingArea'] };
  workflows: { activeWorkflowId: string; lastRuns: Record<string, ProjectWorkflowRunPreview> };
}

/**
 * Which half holds each field of the whole document (`both` for one split inside). Typed against the document, so a
 * field added to it fails to compile until it is placed here, and in `splitProjectDocument`.
 */
export const PROJECT_DOCUMENT_HALVES = {
  canvas: 'both',
  documentSchemaVersion: 'both',
  floatingWidgets: 'workspace',
  id: 'content',
  invocation: 'workspace',
  lastFloatingGeometry: 'workspace',
  layout: 'workspace',
  name: 'content',
  presetWorkingLayouts: 'workspace',
  promptHistory: 'workspace',
  settings: 'workspace',
  widgetGraphs: 'workspace',
  widgetInstances: 'workspace',
  widgetRegions: 'workspace',
  workflows: 'both',
} as const satisfies Record<keyof ProjectDocumentV3, 'both' | 'content' | 'workspace'>;

export interface ProjectHalves {
  content: ProjectContentV4;
  /** Serialized on first read and kept: a caller that only sends the objects never pays for a large document's. */
  readonly contentJson: string;
  workspace: ProjectWorkspaceV4;
  readonly workspaceJson: string;
}

/** Splits a whole document into the two halves the server saves apart. */
export const splitProjectDocument = (document: ProjectDocumentV3): ProjectHalves => {
  const { stagingArea, ...canvas } = document.canvas;
  const lastRuns = new Map<string, ProjectWorkflowRunPreview>();
  const entries = document.workflows.entries.map(({ lastRun, ...entry }) => {
    if (lastRun) {
      lastRuns.set(entry.document.id, lastRun);
    }
    return entry;
  });
  const content: ProjectContentV4 = {
    canvas: { ...canvas, document: { ...canvas.document, selectedLayerId: null } },
    documentSchemaVersion: PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION,
    id: document.id,
    name: document.name,
    workflows: { entries },
  };
  const workspace: ProjectWorkspaceV4 = {
    canvas: { selectedLayerId: document.canvas.document.selectedLayerId, stagingArea },
    documentSchemaVersion: PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION,
    ...(document.floatingWidgets ? { floatingWidgets: document.floatingWidgets } : {}),
    invocation: document.invocation,
    ...(document.lastFloatingGeometry ? { lastFloatingGeometry: document.lastFloatingGeometry } : {}),
    layout: document.layout,
    ...(document.presetWorkingLayouts ? { presetWorkingLayouts: document.presetWorkingLayouts } : {}),
    promptHistory: document.promptHistory,
    settings: document.settings,
    widgetGraphs: document.widgetGraphs,
    widgetInstances: document.widgetInstances,
    widgetRegions: document.widgetRegions,
    // Keyed by workflow id as own properties, whatever the id (`__proto__` through an object literal would not be).
    workflows: { activeWorkflowId: document.workflows.activeWorkflowId, lastRuns: Object.fromEntries(lastRuns) },
  };

  let contentJson: string | undefined;
  let workspaceJson: string | undefined;
  return {
    content,
    get contentJson() {
      return (contentJson ??= JSON.stringify(content));
    },
    workspace,
    get workspaceJson() {
      return (workspaceJson ??= JSON.stringify(workspace));
    },
  };
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const repeats = (ids: unknown[]): boolean => new Set(ids).size !== ids.length;

/**
 * The item ids a lock or presence marker names must each name one item (SF-12): within a workflow, its nodes, its
 * edges and its form elements. Layers are already unique across a canvas document's stacks, checked as it loads.
 * Refused rather than repaired: a repair would have to guess which of two items an edit meant.
 */
export const findRepeatedWorkflowItem = (entries: unknown[]): string | null => {
  for (const entry of entries) {
    const document = isRecord(entry) && isRecord(entry.document) ? entry.document : null;
    if (!document) {
      continue;
    }
    const idsOf = (items: unknown): unknown[] =>
      Array.isArray(items) ? items.map((item) => (isRecord(item) ? item.id : undefined)) : [];
    const form = isRecord(document.form) && isRecord(document.form.elements) ? document.form.elements : {};
    if (repeats(idsOf(document.nodes)) || repeats(idsOf(document.edges)) || repeats(idsOf(Object.values(form)))) {
      return typeof document.id === 'string' ? document.id : '';
    }
  }
  return null;
};

/**
 * Joins the two halves into the whole document the browser works on, which the ordinary migration and load path then
 * reads. A project with no workspace for the caller yet (a 7.1 member, or one never saved) gets `defaults`, the
 * workspace a new project starts with. Content that repeats an item id joins too: a server's must be refused
 * (`joinProjectRecord`), but an archive's is created whole (`toProjectCreation`).
 */
export const joinProjectDocument = (
  content: Record<string, unknown>,
  workspace: Record<string, unknown> | null,
  defaults: () => Omit<ProjectWorkspaceV4, 'documentSchemaVersion'>
): ProjectDocumentMigration => {
  if (workspace !== null && workspace.documentSchemaVersion !== PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION) {
    return { reason: 'The workspace has an unsupported document schema.', status: 'malformed' };
  }
  const canvas = isRecord(content.canvas) ? content.canvas : null;
  const workflows = isRecord(content.workflows) && Array.isArray(content.workflows.entries) ? content.workflows : null;
  if (!canvas || !isRecord(canvas.document) || !workflows) {
    return { reason: 'The project content is missing its canvas or workflows.', status: 'malformed' };
  }
  const entries = workflows.entries as unknown[];

  const fallback = workspace ? null : defaults();
  const own = (workspace ?? fallback) as Record<string, unknown>;
  const ownCanvas = isRecord(own.canvas) ? own.canvas : {};
  const ownWorkflows = isRecord(own.workflows) ? own.workflows : {};
  const lastRuns = isRecord(ownWorkflows.lastRuns) ? ownWorkflows.lastRuns : {};
  const entryIds = entries.map((entry) => (isRecord(entry) && isRecord(entry.document) ? entry.document.id : null));
  // Another member may have deleted the workflow this workspace had open.
  const activeWorkflowId = entryIds.includes(ownWorkflows.activeWorkflowId as string)
    ? ownWorkflows.activeWorkflowId
    : entryIds[0];
  const {
    canvas: _ownCanvas,
    documentSchemaVersion: _workspaceSchema,
    workflows: _ownWorkflows,
    ...workspaceFields
  } = own;
  const { documentSchemaVersion: _contentSchema, ...contentFields } = content;

  return {
    document: {
      ...workspaceFields,
      ...contentFields,
      canvas: {
        ...canvas,
        document: {
          ...canvas.document,
          selectedLayerId: typeof ownCanvas.selectedLayerId === 'string' ? ownCanvas.selectedLayerId : null,
        },
        ...(ownCanvas.stagingArea === undefined ? {} : { stagingArea: ownCanvas.stagingArea }),
      },
      // The whole-document schema the browser works in; the migration boundary reads it as current.
      documentSchemaVersion: PROJECT_DOCUMENT_SCHEMA_VERSION,
      workflows: {
        activeWorkflowId,
        entries: entries.map((entry, index) => {
          const id = entryIds[index] as string;
          // Own properties only: a workflow id such as `constructor` must not read Object.prototype's.
          const run = Object.hasOwn(lastRuns, id) ? lastRuns[id] : undefined;
          return isRecord(entry) && isRecord(run) ? { ...entry, lastRun: run } : entry;
        }),
      },
    },
    status: 'migrated',
  };
};

/**
 * How a document is first saved: its content, under the split floor, with the workspace to save beside it. A workflow
 * that repeats an item id (only a build before the split could write one) would leave content no load accepts, so that
 * project is created whole, as the server would already hold it, and keeps saving whole.
 */
export const toProjectCreation = (
  document: ProjectDocumentV3
):
  | {
      data: ProjectContentV4;
      documentSchemaVersion: typeof PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION;
      workspace: ProjectWorkspaceV4;
    }
  | { data: ProjectDocumentV3; documentSchemaVersion: typeof PROJECT_DOCUMENT_SCHEMA_VERSION; workspace: null } => {
  if (findRepeatedWorkflowItem(document.workflows.entries) !== null) {
    return { data: document, documentSchemaVersion: PROJECT_DOCUMENT_SCHEMA_VERSION, workspace: null };
  }
  const halves = splitProjectDocument(document);
  return {
    data: halves.content,
    documentSchemaVersion: PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION,
    workspace: halves.workspace,
  };
};
