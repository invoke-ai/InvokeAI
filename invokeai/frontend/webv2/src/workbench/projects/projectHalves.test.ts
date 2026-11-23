import type { Project } from '@workbench/projectContracts';

import { addProjectWorkflow, createBlankWorkflowDocument, recordProjectWorkflowRun } from '@workbench/projectWorkflows';
import { asStoredBeforeSplit, createInitialWorkbenchState } from '@workbench/workbenchState.testing';
import { structuralLayouts, workingCopyOf } from '@workbench/workspace/storedLayoutFixtures.testing';
import { describe, expect, it } from 'vitest';

import type { ProjectWithWorkspaceDTO } from './api';

import { serializeProjectDocumentV3, serializeProjectDocumentV3Json } from './projectDocument';
import {
  findRepeatedWorkflowItem,
  joinProjectDocument,
  PROJECT_DOCUMENT_HALVES,
  splitProjectDocument,
} from './projectHalves';
import { joinProjectRecord } from './projectHydration';
import { deserializeProjectDocument } from './syncedPersistence';

const RUN = { completedAt: '2026-10-10T00:00:02.000Z', imageName: 'run.png', submittedAt: '2026-10-10T00:00:01.000Z' };

/** A project whose workspace differs from a fresh one: a second, active workflow with a run, and a staged image. */
const workedOnProject = (): Project => {
  const project = createInitialWorkbenchState().projects[0]!;
  const second = createBlankWorkflowDocument();
  const withSecond = recordProjectWorkflowRun(addProjectWorkflow(project, second), second.id, RUN);
  return {
    ...withSecond,
    canvas: {
      ...withSecond.canvas,
      stagingArea: { ...withSecond.canvas.stagingArea, isVisible: false, selectedImageIndex: 2 },
    },
  };
};

const defaultsOf = (project: Project) => () => {
  const { documentSchemaVersion: _schema, ...workspace } = splitProjectDocument(
    serializeProjectDocumentV3(project)
  ).workspace;
  return workspace;
};

const load = (document: Record<string, unknown>): Project => {
  const loaded = deserializeProjectDocument(document);
  if (loaded.status !== 'loaded') {
    throw new Error(`Expected the document to load, got ${loaded.status}.`);
  }
  return loaded.project;
};

const join = (halves: { content: object; workspace: object | null }, defaults = defaultsOf(workedOnProject())) =>
  joinProjectDocument(
    halves.content as Record<string, unknown>,
    halves.workspace as Record<string, unknown> | null,
    defaults
  );

describe('splitting a project into content and workspace', () => {
  it('rejoins to the same whole document', () => {
    const project = workedOnProject();
    const halves = splitProjectDocument(serializeProjectDocumentV3(project));

    const joined = join({ content: JSON.parse(halves.contentJson), workspace: JSON.parse(halves.workspaceJson) });

    expect(joined.status).toBe('migrated');
    const rejoined = load((joined as { document: Record<string, unknown> }).document);
    // Against the project as loading leaves it: the canonical form sync compares.
    expect(serializeProjectDocumentV3Json(rejoined).documentJson).toBe(
      serializeProjectDocumentV3Json(load(serializeProjectDocumentV3(project))).documentJson
    );
  });

  it('carries every field of the document in one half or the other, in either layout shape', () => {
    const project = workedOnProject();
    const geometry = { heightPx: 300, widthPx: 400, x: 10, y: 20 };
    const current = serializeProjectDocumentV3({
      ...project,
      droppedArrangements: ['edit'],
      lastFloatingGeometry: { gallery: geometry },
      views: { compose: { sizes: { left: 500 } } },
    });
    // A project loaded from a document stored before structure and view split writes that layout back until migrated.
    const stored = serializeProjectDocumentV3(
      asStoredBeforeSplit(project, {
        ...structuralLayouts['floated a widget'],
        heldView: { views: { edit: { sizes: { left: 500 } } }, workspaceId: 'edit' },
        presetWorkingLayouts: [workingCopyOf('edit', { right: { sizePx: 401 } })],
      })
    );
    const fields = new Set<string>();

    for (const document of [current, stored]) {
      const { content, workspace } = splitProjectDocument(document);

      for (const field of Object.keys(document)) {
        const half = PROJECT_DOCUMENT_HALVES[field as keyof typeof PROJECT_DOCUMENT_HALVES];
        const holders = [field in content && 'content', field in workspace && 'workspace'].filter(Boolean);

        fields.add(field);
        expect(holders, field).toEqual(half === 'both' ? ['content', 'workspace'] : [half]);
      }
    }
    expect([...fields].sort()).toEqual(Object.keys(PROJECT_DOCUMENT_HALVES).sort());
    expect(splitProjectDocument(current).workspace.documentSchemaVersion).toBe(5);
    expect(splitProjectDocument(stored).workspace.documentSchemaVersion).toBe(4);
  });

  it('keeps how one member works out of the shared content', () => {
    const project = workedOnProject();
    const { content, workspace } = splitProjectDocument(serializeProjectDocumentV3(project));

    expect(Object.keys(content).sort()).toEqual(['canvas', 'documentSchemaVersion', 'id', 'name', 'workflows']);
    expect('stagingArea' in content.canvas).toBe(false);
    expect(content.canvas.document.selectedLayerId).toBeNull();
    expect(content.workflows.entries.some((entry) => 'lastRun' in entry)).toBe(false);
    expect(Object.values(workspace.workflows.lastRuns)).toEqual([RUN]);
    expect(workspace.canvas.stagingArea.selectedImageIndex).toBe(2);
  });

  it.each(['__proto__', 'constructor', 'toString'])('round-trips the last run of a workflow with id %s', (id) => {
    const project = createInitialWorkbenchState().projects[0]!;
    const workflow = { ...createBlankWorkflowDocument(), id };
    const ran = recordProjectWorkflowRun(addProjectWorkflow(project, workflow), id, RUN);
    const halves = splitProjectDocument(serializeProjectDocumentV3(ran));

    const joined = join({ content: JSON.parse(halves.contentJson), workspace: JSON.parse(halves.workspaceJson) });

    expect(Object.hasOwn(halves.workspace.workflows.lastRuns, id)).toBe(true);
    const rejoined = load((joined as { document: Record<string, unknown> }).document);
    expect(rejoined.workflows.entries.find((entry) => entry.document.id === id)?.lastRun).toEqual(RUN);
    expect(rejoined.workflows.entries.filter((entry) => entry.lastRun)).toHaveLength(1);
  });

  it('changes only the half an edit belongs to', () => {
    const project = workedOnProject();
    const before = splitProjectDocument(serializeProjectDocumentV3(project));

    const relaid = splitProjectDocument(
      serializeProjectDocumentV3({ ...project, views: { compose: { fronts: { center: 'gallery:center' } } } })
    );
    const renamed = splitProjectDocument(serializeProjectDocumentV3({ ...project, name: 'Renamed' }));

    expect(relaid.contentJson).toBe(before.contentJson);
    expect(relaid.workspaceJson).not.toBe(before.workspaceJson);
    expect(renamed.workspaceJson).toBe(before.workspaceJson);
    expect(renamed.contentJson).not.toBe(before.contentJson);
  });
});

describe('joining a project for a member', () => {
  it('starts a member with no workspace on the defaults, on the first workflow', () => {
    const project = workedOnProject();
    const { content } = splitProjectDocument(serializeProjectDocumentV3(project));

    const joined = join({ content, workspace: null });

    const loaded = load((joined as { document: Record<string, unknown> }).document);
    expect(loaded.workflows.activeWorkflowId).toBe(project.workflows.entries[0]!.document.id);
    expect(loaded.workflows.entries.some((entry) => entry.lastRun)).toBe(false);
    expect(loaded.name).toBe(project.name);
    // A member's first open is a new workspace: Auto-switch starts In progress, not a stored project's Off.
    expect(loaded.settings.viewerAutoSwitch).toBe('progress');
  });

  it('keeps the Auto-switch mode a workspace saved, across a split and a join', () => {
    const project = workedOnProject();
    const { content, workspace } = splitProjectDocument(
      serializeProjectDocumentV3({ ...project, settings: { ...project.settings, viewerAutoSwitch: 'latest' } })
    );

    const joined = join({ content, workspace });

    expect(load((joined as { document: Record<string, unknown> }).document).settings.viewerAutoSwitch).toBe('latest');
  });

  it('falls back to the first workflow when the open one was deleted meanwhile', () => {
    const project = workedOnProject();
    const { content, workspace } = splitProjectDocument(serializeProjectDocumentV3(project));

    const joined = join({
      content,
      workspace: { ...workspace, workflows: { activeWorkflowId: 'gone', lastRuns: {} } },
    });

    expect(load((joined as { document: Record<string, unknown> }).document).workflows.activeWorkflowId).toBe(
      project.workflows.entries[0]!.document.id
    );
  });

  it('refuses a server’s content that repeats an item id within a workflow, which a lock could not tell apart', () => {
    const project = workedOnProject();
    const { content, workspace } = splitProjectDocument(serializeProjectDocumentV3(project));
    const [first, ...rest] = content.workflows.entries;
    const node = { id: 'node-1', type: 'notes' };
    const repeated = { ...first!, document: { ...first!.document, nodes: [node, node] } };
    const elsewhere = rest.map((entry) => ({ ...entry, document: { ...entry.document, nodes: [node] } }));
    const record = (entries: unknown[]) =>
      ({
        data: { ...content, workflows: { entries } },
        workspace: { data: workspace, revision: 1, updated_at: '' },
      }) as unknown as ProjectWithWorkspaceDTO;

    expect(joinProjectRecord(record([repeated, ...rest])).status).toBe('malformed');
    // An archive's joins: importing creates it whole.
    expect(join({ content: { ...content, workflows: { entries: [repeated, ...rest] } }, workspace }).status).toBe(
      'migrated'
    );
    // Workflow duplication keeps node ids, so two workflows may share one.
    expect(
      joinProjectRecord(record([{ ...first!, document: { ...first!.document, nodes: [node] } }, ...elsewhere])).status
    ).toBe('migrated');
  });

  it('finds repeated edge and form element ids too', () => {
    const { content } = splitProjectDocument(serializeProjectDocumentV3(workedOnProject()));
    const [first, ...rest] = content.workflows.entries;
    const edge = { id: 'edge-1', source: 'a', sourceHandle: 'value', target: 'b', targetHandle: 'value' };
    const element = { data: { children: [], layout: 'column' }, id: 'container-1', type: 'container' };
    const repeating = (document: object) => [{ ...first!, document: { ...first!.document, ...document } }, ...rest];

    expect(findRepeatedWorkflowItem(repeating({ edges: [edge, edge] }))).toBe(first!.document.id);
    expect(
      findRepeatedWorkflowItem(repeating({ form: { elements: { a: element, b: element }, rootElementId: 'a' } }))
    ).toBe(first!.document.id);
    expect(findRepeatedWorkflowItem(content.workflows.entries)).toBeNull();
  });

  it('refuses a workspace of another document schema', () => {
    const { content, workspace } = splitProjectDocument(serializeProjectDocumentV3(workedOnProject()));

    expect(join({ content, workspace: { ...workspace, documentSchemaVersion: 6 } }).status).toBe('malformed');
    expect(join({ content, workspace: { ...workspace, documentSchemaVersion: 3 } }).status).toBe('malformed');
  });
});
