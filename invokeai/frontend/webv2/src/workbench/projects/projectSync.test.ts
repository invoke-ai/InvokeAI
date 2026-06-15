import { describe, expect, it } from 'vitest';

import { createRecoveredDocument, deserializeProjectDocument, serializeProjectDocument } from './syncedPersistence';
import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState';
import type { Project } from '@workbench/types';

const getProject = (overrides: Partial<Project> = {}): Project => {
  const state = createInitialWorkbenchState();

  return { ...state.projects[0], ...overrides };
};

describe('project document serialization', () => {
  it('strips undo/redo history and restores it empty on deserialize', () => {
    const project = getProject();

    project.undoRedo.past.push({
      createdAt: 'now',
      id: 'undo-1',
      label: 'test',
      project: {
        canvas: project.canvas,
        invocation: project.invocation,
        layout: project.layout,
        projectGraph: project.projectGraph,
        widgetGraphs: {},
        widgetRegions: project.widgetRegions,
        widgetStates: project.widgetStates,
      },
    });

    const document = serializeProjectDocument(project);

    expect('undoRedo' in document).toBe(false);

    const roundTripped = deserializeProjectDocument(document);

    expect(roundTripped).not.toBeNull();
    expect(roundTripped?.undoRedo).toEqual({ future: [], past: [] });
    expect(roundTripped?.id).toBe(project.id);
    expect(roundTripped?.widgetStates).toEqual(project.widgetStates);
  });

  it('rejects documents that do not look like projects', () => {
    expect(deserializeProjectDocument({})).toBeNull();
    expect(deserializeProjectDocument({ id: 'x' })).toBeNull();
    expect(deserializeProjectDocument({ id: 'x', layout: null, name: 'y' })).toBeNull();
  });
});

describe('createRecoveredDocument', () => {
  it('keys the fork to the original and stamps the recovery time', () => {
    const project = getProject({ name: 'My Project' });
    const { recoveredDocument, recoveredId, recoveredName } = createRecoveredDocument(
      project,
      serializeProjectDocument(project)
    );

    expect(recoveredDocument.recoveryOf).toBe(project.id);
    expect(recoveredId.startsWith(`${project.id}-recovered-`)).toBe(true);
    expect(recoveredName).toBe('My Project (recovered)');
    expect(typeof recoveredDocument.recoveredAt).toBe('string');
  });

  it('collapses recovery chains to the root and never stacks name suffixes', () => {
    const root = getProject({ name: 'My Project' });
    const recovery = getProject({
      id: `${root.id}-recovered-abc`,
      name: 'My Project (recovered)',
      recoveryOf: root.id,
    });

    const { recoveredDocument, recoveredName } = createRecoveredDocument(recovery, serializeProjectDocument(recovery));

    expect(recoveredDocument.recoveryOf).toBe(root.id);
    expect(recoveredName).toBe('My Project (recovered)');
  });

  it('cleans up legacy stacked suffixes', () => {
    const project = getProject({ name: 'Project Name #1 (Recovered) (Recovered)' });
    const { recoveredName } = createRecoveredDocument(project, serializeProjectDocument(project));

    expect(recoveredName).toBe('Project Name #1 (recovered)');
  });
});

describe('openProject', () => {
  it('appends the hydrated project and makes it active', () => {
    const state = createInitialWorkbenchState();
    const opened = getProject({ id: 'project-from-library', name: 'Reopened' });

    const next = workbenchReducer(state, { project: opened, type: 'openProject' });

    expect(next.projects.map((project) => project.id)).toEqual([state.projects[0].id, opened.id]);
    expect(next.activeProjectId).toBe(opened.id);
  });

  it('focuses an already-open project instead of duplicating it', () => {
    const state = createInitialWorkbenchState();
    const existing = state.projects[0];
    const background = getProject({ id: 'project-background' });
    const withTwo = workbenchReducer(state, { project: background, type: 'openProject' });

    const next = workbenchReducer(withTwo, { project: existing, type: 'openProject' });

    expect(next.projects).toHaveLength(2);
    expect(next.activeProjectId).toBe(existing.id);
  });
});

describe('renameProject', () => {
  it('renames the target project and ignores blank names', () => {
    const state = createInitialWorkbenchState();
    const target = state.projects[0];

    const renamed = workbenchReducer(state, { name: '  New Name  ', projectId: target.id, type: 'renameProject' });

    expect(renamed.projects[0].name).toBe('New Name');

    const blank = workbenchReducer(renamed, { name: '   ', projectId: target.id, type: 'renameProject' });

    expect(blank.projects[0].name).toBe('New Name');
  });
});

describe('reconcileProjectConflict', () => {
  it('adopts the server version and continues local work in the recovered fork', () => {
    const state = createInitialWorkbenchState();
    const original = state.projects[0];
    const serverProject = getProject({ id: original.id, name: 'Server version' });
    const recoveredProject = getProject({ id: `${original.id}-recovered-abc`, name: 'Server version (recovered)' });
    const withActiveOriginal = { ...state, activeProjectId: original.id };

    const next = workbenchReducer(withActiveOriginal, {
      projectId: original.id,
      recoveredProject,
      serverProject,
      type: 'reconcileProjectConflict',
    });

    const ids = next.projects.map((project) => project.id);

    expect(ids).toContain(original.id);
    expect(ids).toContain(recoveredProject.id);
    expect(next.projects.find((project) => project.id === original.id)?.name).toBe('Server version');
    // The user keeps looking at their own latest edits.
    expect(next.activeProjectId).toBe(recoveredProject.id);
    expect(next.notifications[0]?.title).toBe('Project recovered');
  });

  it('leaves the active project alone when the conflicted project is in the background', () => {
    const state = createInitialWorkbenchState();
    const first = state.projects[0];
    const second = getProject({ id: 'project-background-test' });
    const withActiveSecond = { ...state, activeProjectId: second.id, projects: [...state.projects, second] };

    const next = workbenchReducer(withActiveSecond, {
      projectId: first.id,
      recoveredProject: getProject({ id: `${first.id}-recovered-abc`, name: 'Recovered' }),
      serverProject: getProject({ id: first.id, name: 'Server version' }),
      type: 'reconcileProjectConflict',
    });

    expect(next.activeProjectId).toBe(second.id);
  });
});
