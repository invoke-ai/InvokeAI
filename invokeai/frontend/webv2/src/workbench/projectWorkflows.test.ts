import type { InvocationTemplate, ProjectGraphState } from '@features/workflow/contracts';

import { buildInvocationNode, createProjectGraph } from '@features/workflow/utility';
import { describe, expect, it, vi } from 'vitest';

import {
  addProjectWorkflow,
  applyProjectWorkflowAction,
  createProjectWorkflowCollection,
  duplicateProjectWorkflow,
  getActiveProjectGraph,
  isPlaceholderProjectWorkflow,
  migrateProjectGraphToCollection,
  normalizeProjectWorkflowCollection,
  recordProjectWorkflowRun,
  redoProjectWorkflow,
  removeProjectWorkflow,
  replaceProjectWorkflowDocument,
  selectProjectWorkflow,
  setProjectWorkflowSource,
  stripProjectWorkflowSources,
  undoProjectWorkflow,
  WORKFLOW_HISTORY_LIMIT,
  type ProjectWorkflowSession,
} from './projectWorkflows';

const template: InvocationTemplate = {
  category: 'test',
  classification: 'stable',
  description: '',
  inputs: {},
  nodePack: 'invokeai',
  outputs: {},
  outputType: 'noise_output',
  tags: [],
  title: 'Noise',
  type: 'noise',
  useCache: true,
  version: '1.0.0',
};

const session = (...documents: ProjectGraphState[]): ProjectWorkflowSession => ({
  workflowHistories: {},
  workflows: { activeWorkflowId: documents[0]!.id, entries: documents.map((document) => ({ document })) },
});

const ids = (project: ProjectWorkflowSession) => project.workflows.entries.map((entry) => entry.document.id);

const addNode = (project: ProjectWorkflowSession, workflowId: string, nodeId = `node-${Math.random()}`) =>
  applyProjectWorkflowAction(project, workflowId, {
    node: { ...buildInvocationNode(template, { x: 0, y: 0 }), id: nodeId },
    type: 'addNode',
  }).project;

describe('project workflow collection', () => {
  it('keeps a blank that was edited and undone: its history is authored work', () => {
    let project = session(createProjectGraph('blank'));

    project = addNode(project, 'blank', 'n1');
    project = applyProjectWorkflowAction(project, 'blank', { nodeIds: ['n1'], type: 'removeNodes' }).project;
    expect(getActiveProjectGraph(project).nodes).toEqual([]);

    const next = addProjectWorkflow(project, createProjectGraph('template'), { reusePlaceholder: true });

    expect(ids(next)).toEqual(['blank', 'template']);
    expect(next.workflowHistories.blank?.past).toHaveLength(2);
  });

  it('opens a template over the untouched blank a project starts with, but never over authored work', () => {
    const blank = createProjectGraph('blank');
    const template1 = createProjectGraph('template-1', 'Template');
    const opened = addProjectWorkflow(session(blank), template1, {
      reusePlaceholder: true,
      source: { libraryWorkflowId: 'lib-1', revision: 3 },
    });

    expect(ids(opened)).toEqual(['template-1']);
    expect(opened.workflows.activeWorkflowId).toBe('template-1');
    expect(opened.workflows.entries[0]?.source).toEqual({ libraryWorkflowId: 'lib-1', revision: 3 });

    const authored = addNode(session(blank), 'blank');
    const beside = addProjectWorkflow(authored, template1, { reusePlaceholder: true });

    expect(ids(beside)).toEqual(['blank', 'template-1']);

    const renamed = applyProjectWorkflowAction(session(blank), 'blank', {
      patch: { name: 'Mine' },
      type: 'setMetadata',
    }).project;

    expect(isPlaceholderProjectWorkflow(renamed.workflows.entries[0]!)).toBe(false);
    expect(ids(addProjectWorkflow(renamed, template1, { reusePlaceholder: true }))).toEqual(['blank', 'template-1']);
  });

  it('refuses a document id the project already owns', () => {
    const project = session(createProjectGraph('a'));

    expect(() => addProjectWorkflow(project, createProjectGraph('a'))).toThrow(/already owns/);
  });

  it('selects a neighbour on removal and leaves a fresh blank when the last workflow goes', () => {
    let project = session(createProjectGraph('a'), createProjectGraph('b'), createProjectGraph('c'));

    project = selectProjectWorkflow(project, 'b');
    project = addNode(project, 'b');
    expect(project.workflowHistories.b?.past).toHaveLength(1);

    project = removeProjectWorkflow(project, 'b');
    expect(ids(project)).toEqual(['a', 'c']);
    expect(project.workflows.activeWorkflowId).toBe('c');
    expect(project.workflowHistories.b).toBeUndefined();

    project = removeProjectWorkflow(project, 'c');
    expect(project.workflows.activeWorkflowId).toBe('a');

    project = removeProjectWorkflow(project, 'a');
    expect(project.workflows.entries).toHaveLength(1);
    expect(project.workflows.entries[0]?.document.nodes).toEqual([]);
    expect(project.workflows.activeWorkflowId).toBe(project.workflows.entries[0]?.document.id);
    expect(project.workflows.entries[0]?.document.id).not.toBe('a');
  });

  it('removing an inactive workflow keeps the active selection', () => {
    const project = removeProjectWorkflow(session(createProjectGraph('a'), createProjectGraph('b')), 'b');

    expect(project.workflows.activeWorkflowId).toBe('a');
  });

  it('duplicates under a fresh id, beside the original, carrying the source but not the run preview', () => {
    let project = session(createProjectGraph('a', 'Original'), createProjectGraph('b'));

    project = setProjectWorkflowSource(project, 'a', { libraryWorkflowId: 'lib-1', revision: 2 });
    project = recordProjectWorkflowRun(project, 'a', { completedAt: 't', imageName: 'out.png', submittedAt: 't' });
    project = duplicateProjectWorkflow(project, 'a', 'copy', (name) => `${name} copy`);

    expect(ids(project)).toEqual(['a', 'copy', 'b']);
    expect(project.workflows.activeWorkflowId).toBe('copy');
    expect(project.workflows.entries[1]).toMatchObject({
      document: { name: 'Original copy' },
      source: { libraryWorkflowId: 'lib-1', revision: 2 },
    });
    expect(project.workflows.entries[1]?.lastRun).toBeUndefined();
  });

  it('records the newest submitted successful run, not the run that finishes last', () => {
    let project = session(createProjectGraph('a'), createProjectGraph('b'));

    project = recordProjectWorkflowRun(project, 'a', {
      completedAt: '2026-01-01T00:00:10.000Z',
      imageName: 'newer.png',
      submittedAt: '2026-01-01T00:00:02.000Z',
    });
    project = recordProjectWorkflowRun(project, 'a', {
      completedAt: '2026-01-01T00:00:12.000Z',
      imageName: 'older.png',
      submittedAt: '2026-01-01T00:00:01.000Z',
    });

    expect(project.workflows.entries[0]?.lastRun?.imageName).toBe('newer.png');
    expect(project.workflows.entries[1]?.lastRun).toBeUndefined();
    expect(recordProjectWorkflowRun(project, 'missing', { completedAt: 't', imageName: 'x', submittedAt: 't' })).toBe(
      project
    );
  });
});

describe('per-workflow edit history', () => {
  it('keeps each workflow its own history and shares structure instead of cloning', () => {
    let project = session(createProjectGraph('a'), createProjectGraph('b'));
    const untouched = project.workflows.entries[1];
    const before = getActiveProjectGraph(project);

    project = addNode(project, 'a', 'n1');

    expect(project.workflows.entries[1]).toBe(untouched);
    expect(project.workflowHistories.a?.past[0]?.document).toBe(before);
    expect(project.workflowHistories.b).toBeUndefined();

    project = undoProjectWorkflow(project, 'a');
    expect(getActiveProjectGraph(project)).toBe(before);
    expect(project.workflowHistories.a?.future).toHaveLength(1);

    project = redoProjectWorkflow(project, 'a');
    expect(getActiveProjectGraph(project).nodes.map((node) => node.id)).toEqual(['n1']);
    expect(undoProjectWorkflow(project, 'b')).toBe(project);
  });

  it('undoing an edit leaves the source and run metadata where they were', () => {
    let project = session(createProjectGraph('a'));

    project = addNode(project, 'a', 'n1');
    project = setProjectWorkflowSource(project, 'a', { libraryWorkflowId: 'lib-1', revision: 5 });
    project = undoProjectWorkflow(project, 'a');

    expect(project.workflows.entries[0]).toMatchObject({
      document: { nodes: [] },
      source: { libraryWorkflowId: 'lib-1', revision: 5 },
    });
  });

  it('folds a stream of same-key edits into one step and separates them after a pause', () => {
    vi.useFakeTimers({ now: new Date('2026-06-10T00:00:00.000Z') });

    try {
      let project = addNode(session(createProjectGraph('a')), 'a', 'n1');
      const type = (value: string) =>
        applyProjectWorkflowAction(project, 'a', { patch: { notes: value }, type: 'setMetadata' }).project;

      project = type('h');
      project = type('he');
      project = type('hel');
      expect(project.workflowHistories.a?.past).toHaveLength(2);

      vi.advanceTimersByTime(2000);
      project = type('hell');
      expect(project.workflowHistories.a?.past).toHaveLength(3);

      project = undoProjectWorkflow(project, 'a');
      expect(getActiveProjectGraph(project).notes).toBe('hel');
      project = undoProjectWorkflow(project, 'a');
      expect(getActiveProjectGraph(project).notes).toBe('');
    } finally {
      vi.useRealTimers();
    }
  });

  it('evicts the oldest entries across workflows once the project holds forty', () => {
    let project = session(createProjectGraph('a'), createProjectGraph('b'));

    for (let index = 0; index < 30; index += 1) {
      project = addNode(project, 'a', `a-${index}`);
    }
    for (let index = 0; index < 15; index += 1) {
      project = addNode(project, 'b', `b-${index}`);
    }

    const total = Object.values(project.workflowHistories).reduce((count, history) => count + history.past.length, 0);

    expect(total).toBe(WORKFLOW_HISTORY_LIMIT);
    // The five oldest steps belonged to `a`; `b`'s newer steps are all still there.
    expect(project.workflowHistories.a?.past).toHaveLength(25);
    expect(project.workflowHistories.b?.past).toHaveLength(15);
  });

  it('counts redo entries against the same budget and lets the oldest of them go first', () => {
    let project = session(createProjectGraph('a'), createProjectGraph('b'));

    for (let index = 0; index < 20; index += 1) {
      project = addNode(project, 'a', `a-${index}`);
    }
    for (let index = 0; index < 20; index += 1) {
      project = undoProjectWorkflow(project, 'a');
    }

    expect(project.workflowHistories.a).toMatchObject({ future: expect.any(Array), past: [] });
    expect(project.workflowHistories.a?.future).toHaveLength(20);

    for (let index = 0; index < 25; index += 1) {
      project = addNode(project, 'b', `b-${index}`);
    }

    const total = Object.values(project.workflowHistories).reduce(
      (count, history) => count + history.past.length + history.future.length,
      0
    );

    expect(total).toBe(WORKFLOW_HISTORY_LIMIT);
    // The redo entries were created before any of b's steps, so five of them went; the nearest redo survives.
    expect(project.workflowHistories.a?.future).toHaveLength(15);
    expect(project.workflowHistories.b?.past).toHaveLength(25);
    expect(getActiveProjectGraph(redoProjectWorkflow(project, 'a')).nodes.map((node) => node.id)).toEqual(['a-0']);
  });
});

describe('replacing a copy with its library version', () => {
  it('keeps the copy’s id and name, moves its source, and undoes to the previous graph in one step', () => {
    const copy = { ...createProjectGraph('copy'), name: 'My tweaks' };
    let project: ProjectWorkflowSession = {
      workflowHistories: {},
      workflows: {
        activeWorkflowId: 'copy',
        entries: [{ document: copy, source: { libraryWorkflowId: 'lib', revision: 1 } }],
      },
    };

    project = addNode(project, 'copy', 'mine');

    const authored = getActiveProjectGraph(project);
    const library = { ...createProjectGraph('parsed-elsewhere'), name: 'Library name' };
    const libraryWithNode = applyProjectWorkflowAction(session(library), library.id, {
      node: { ...buildInvocationNode(template, { x: 10, y: 10 }), id: 'theirs' },
      type: 'addNode',
    }).project;

    project = replaceProjectWorkflowDocument(project, 'copy', getActiveProjectGraph(libraryWithNode), {
      label: 'Replace',
      source: { libraryWorkflowId: 'lib', revision: 4 },
    });

    const replaced = project.workflows.entries[0]!;

    expect(replaced.document).toMatchObject({ id: 'copy', name: 'My tweaks' });
    expect(replaced.document.nodes.map((node) => node.id)).toEqual(['theirs']);
    expect(replaced.source).toEqual({ libraryWorkflowId: 'lib', revision: 4 });

    project = undoProjectWorkflow(project, 'copy');

    // The graph and the revision it is based on travel together, so a later update still meets the conflict check.
    expect(getActiveProjectGraph(project)).toBe(authored);
    expect(project.workflows.entries[0]!.source).toEqual({ libraryWorkflowId: 'lib', revision: 1 });

    project = redoProjectWorkflow(project, 'copy');

    expect(getActiveProjectGraph(project).nodes.map((node) => node.id)).toEqual(['theirs']);
    expect(project.workflows.entries[0]!.source).toEqual({ libraryWorkflowId: 'lib', revision: 4 });
  });

  it('leaves the project alone when the copy is gone', () => {
    const project = session(createProjectGraph('a'));

    expect(
      replaceProjectWorkflowDocument(project, 'missing', createProjectGraph('b'), {
        label: 'Replace',
        source: { libraryWorkflowId: 'lib', revision: 1 },
      })
    ).toBe(project);
  });
});

describe('persisted collections', () => {
  it('migrates a single graph into the first, active workflow and turns a binding into an unknown-revision source', () => {
    const graph = { ...createProjectGraph('legacy'), libraryWorkflowId: 'lib-9' };
    const collection = migrateProjectGraphToCollection(graph)!;

    expect(collection.activeWorkflowId).toBe('legacy');
    expect(collection.entries[0]?.source).toEqual({ libraryWorkflowId: 'lib-9', revision: null });
    expect('libraryWorkflowId' in collection.entries[0]!.document).toBe(false);
    expect(migrateProjectGraphToCollection(undefined)?.entries[0]?.source).toBeUndefined();
  });

  it('refuses a damaged or unsupported legacy graph instead of replacing it, but blanks the Phase-1 placeholder', () => {
    const authored = createProjectGraph('legacy');

    expect(migrateProjectGraphToCollection({ ...authored, nodes: undefined })).toBeNull();
    expect(migrateProjectGraphToCollection({ ...authored, form: { elements: {} } })).toBeNull();
    // An authored graph from a newer client is not this build's to rewrite.
    expect(migrateProjectGraphToCollection({ ...authored, version: 99 })).toBeNull();
    expect(migrateProjectGraphToCollection({ id: 'old', nodes: [{ id: 'n1' }], version: 1 })).toBeNull();
    // The placeholder never held content: it becomes a blank under its own id.
    const placeholder = migrateProjectGraphToCollection({ edges: [], id: 'phase-1', nodes: [], version: 1 });

    expect(placeholder?.entries[0]?.document).toMatchObject({ id: 'phase-1', nodes: [], version: 2 });
    expect(migrateProjectGraphToCollection({ id: 'phase-1' })?.entries[0]?.document.id).toBe('phase-1');
  });

  it('refuses a collection whose authored document is damaged, never a blank in its place', () => {
    const good = createProjectWorkflowCollection(createProjectGraph('a'));
    const [entry] = good.entries;
    const damaged = (document: Record<string, unknown>) =>
      normalizeProjectWorkflowCollection({ ...good, entries: [{ ...entry, document }] });

    // An unsupported document version is authored content this build cannot read.
    expect(damaged({ ...entry!.document, version: 99 })).toBeNull();
    expect(damaged({ ...entry!.document, nodes: undefined })).toBeNull();
    expect(damaged({ ...entry!.document, nodes: [{ id: 'n1' }] })).toBeNull();
    expect(damaged({ ...entry!.document, form: undefined })).toBeNull();
    expect(damaged({ ...entry!.document, form: { elements: {}, rootElementId: 'missing' } })).toBeNull();
    // The structures the editor and the linear form read must be there, not just ids and types.
    const root = entry!.document.form.rootElementId;

    expect(
      damaged({
        ...entry!.document,
        form: { elements: { [root]: { id: root, type: 'container' } }, rootElementId: root },
      })
    ).toBeNull();
    expect(
      damaged({
        ...entry!.document,
        form: {
          elements: {
            [root]: { data: { children: ['f'], layout: 'column' }, id: root, type: 'container' },
            f: { data: {}, id: 'f', type: 'node-field' },
          },
          rootElementId: root,
        },
      })
    ).toBeNull();
    expect(
      damaged({
        ...entry!.document,
        form: { elements: { [root]: { data: { content: 'x' }, id: root, type: 'heading' } }, rootElementId: root },
      })
    ).toBeNull();
    expect(
      damaged({
        ...entry!.document,
        nodes: [{ data: { label: 'no inputs' }, id: 'n1', position: { x: 0, y: 0 }, type: 'invocation' }],
      })
    ).toBeNull();
    // Missing metadata strings are filled in; the authored graph is kept as it is.
    const { description: _description, ...withoutDescription } = entry!.document;
    const filled = damaged(withoutDescription);

    expect(filled?.entries[0]?.document.description).toBe('');
    expect(filled?.entries[0]?.document.nodes).toBe(entry!.document.nodes);
  });

  it('accepts a well-formed collection and refuses one that would lose a workflow', () => {
    const good = createProjectWorkflowCollection(createProjectGraph('a'));

    expect(normalizeProjectWorkflowCollection(good)).toEqual(good);
    expect(normalizeProjectWorkflowCollection({ ...good, activeWorkflowId: 'missing' })).toBeNull();
    expect(normalizeProjectWorkflowCollection({ activeWorkflowId: 'a', entries: [] })).toBeNull();
    expect(
      normalizeProjectWorkflowCollection({ ...good, entries: [...good.entries, { document: { id: 'a' } }] })
    ).toBeNull();
    expect(normalizeProjectWorkflowCollection({ ...good, entries: [{ document: null }] })).toBeNull();
    expect(
      normalizeProjectWorkflowCollection({ ...good, entries: [{ ...good.entries[0], source: { revision: 1 } }] })
    ).toBeNull();
    // A run preview is decoration: a damaged one is dropped, never the project.
    expect(
      normalizeProjectWorkflowCollection({
        ...good,
        entries: [{ ...good.entries[0], lastRun: { imageName: 'a.png', submittedAt: 'now' } }],
      })?.entries[0]
    ).toEqual({ document: good.entries[0]!.document });
    expect(
      normalizeProjectWorkflowCollection({
        ...good,
        entries: [{ ...good.entries[0], source: { libraryWorkflowId: 'lib', revision: 'x' } }],
      })?.entries[0]?.source
    ).toEqual({ libraryWorkflowId: 'lib', revision: null });
  });

  it('strips write provenance for portable documents', () => {
    const collection = createProjectWorkflowCollection(createProjectGraph('a'));

    expect(stripProjectWorkflowSources(collection)).toBe(collection);
    expect(
      stripProjectWorkflowSources({
        ...collection,
        entries: [{ ...collection.entries[0]!, source: { libraryWorkflowId: 'lib', revision: 1 } }],
      }).entries[0]
    ).toEqual({ document: collection.entries[0]!.document });
  });
});
