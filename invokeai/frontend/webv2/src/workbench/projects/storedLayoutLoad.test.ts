import type { Project } from '@workbench/projectContracts';
import type { StoredProjectLayout } from '@workbench/workspace/migrateProjectLayout';

import { getLayoutPreset } from '@workbench/layoutPresets';
import {
  asStoredBeforeSplit,
  createInitialWorkbenchState,
  getActiveProject,
  getLayoutOf,
  workbenchReducer,
} from '@workbench/workbenchState.testing';
import { registryAvailability } from '@workbench/workspace/layoutFixtures.testing';
import { resolveLayout } from '@workbench/workspace/resolveLayout';
import { viewOnlyLayouts } from '@workbench/workspace/storedLayoutFixtures.testing';
import { normalizeStructure } from '@workbench/workspace/structure';
import { describe, expect, it } from 'vitest';

import type { ProjectWithWorkspaceDTO } from './api';

import { serializeProjectDocument, serializeProjectDocumentV3 } from './projectDocument';
import { splitProjectDocument } from './projectHalves';
import {
  deserializeProjectDocument,
  deserializeProjectRecord,
  joinProjectHalves,
  joinProjectRecord,
} from './projectHydration';

const loaded = (result: ReturnType<typeof deserializeProjectDocument>): Project => {
  if (result.status !== 'loaded') {
    throw new Error(`Expected the document to load, got ${result.status}.`);
  }
  return result.project;
};

const storedProject = (stored: StoredProjectLayout): Project =>
  asStoredBeforeSplit(createInitialWorkbenchState().projects[0]!, stored);

const splitRecordOf = (document: Record<string, unknown>, project: Project): ProjectWithWorkspaceDTO => {
  const halves = splitProjectDocument(document as unknown as ReturnType<typeof serializeProjectDocumentV3>);

  return {
    board_id: 'board',
    created_at: '2026-01-01T00:00:00Z',
    data: halves.content as unknown as Record<string, unknown>,
    minimum_canvas_schema_version: 3,
    minimum_document_schema_version: 4,
    name: project.name,
    project_id: project.id,
    revision: 1,
    updated_at: '2026-01-01T00:00:00Z',
    workspace: {
      data: halves.workspace as unknown as Record<string, unknown>,
      revision: 1,
      updated_at: '2026-01-01T00:00:00Z',
    },
    workspace_updated_at: '2026-01-01T00:00:00Z',
  };
};

/** Each way a stored document reaches the editor, as the project it loads to. */
const loadPaths = (project: Project): Record<string, Project> => {
  const whole = serializeProjectDocumentV3(project) as unknown as Record<string, unknown>;
  const record = splitRecordOf(whole, project);
  const archive = splitProjectDocument(serializeProjectDocumentV3(project));
  const joinedArchive = joinProjectHalves(
    archive.content as unknown as Record<string, unknown>,
    archive.workspace as unknown as Record<string, unknown>
  );
  const duplicateSource = joinProjectRecord(record);

  if (joinedArchive.status !== 'migrated' || duplicateSource.status !== 'migrated') {
    throw new Error('Expected the halves to join.');
  }

  return {
    'a .invk archive': loaded(deserializeProjectDocument(joinedArchive.document)),
    // A copy is written back as the source stored it, and migrates when the editor opens it.
    'a duplicate': loaded(
      deserializeProjectDocument(serializeProjectDocument(loaded(deserializeProjectDocument(duplicateSource.document))))
    ),
    'a local draft or journal entry': loaded(deserializeProjectDocument(whole)),
    'a server record': loaded(deserializeProjectRecord(record)),
  };
};

const open = (project: Project) => workbenchReducer(createInitialWorkbenchState(), { project, type: 'openProject' });

/** Each reducer step that settles a stored project against the account, as the state it leaves active. */
const settlePaths = (project: Project): Record<string, ReturnType<typeof open>> => {
  const opened = open(project);
  const initial = createInitialWorkbenchState();
  // Cross-tab sync, or a conflict resolved to the server's workspace, replaces the open project with the server's.
  const replaced = workbenchReducer(
    { ...initial, activeProjectId: project.id, projects: [{ ...initial.projects[0]!, id: project.id }] },
    { project, projectId: project.id, type: 'replaceProjectFromServer' }
  );
  // Save as new: the server's copy under a new id, when this tab no longer holds the source.
  const retargeted = workbenchReducer(initial, {
    boardId: 'board',
    name: 'Copy',
    project,
    projectId: 'gone',
    sourceName: project.name,
    targetProjectId: project.id,
    type: 'retargetProject',
  });

  return {
    open: opened,
    'replace from the server': replaced,
    'save as new': { ...retargeted, activeProjectId: project.id },
  };
};

describe('documents stored before structure and view split', () => {
  it('shows what the project showed through every load path, when it changed only how it looked', () => {
    for (const [name, stored] of Object.entries(viewOnlyLayouts)) {
      const arrangement = normalizeStructure({
        ...(stored.floatingWidgets ? { floatingWidgets: stored.floatingWidgets } : {}),
        layout: stored.layout,
        widgetInstances: Object.fromEntries(
          Object.entries(stored.widgetInstances).map(([id, { typeId }]) => [id, { id, typeId }])
        ),
        widgetRegions: stored.widgetRegions,
      });
      const before = resolveLayout(arrangement, undefined, registryAvailability);
      const workspaceId = stored.layout.presetId as 'compose' | 'edit';

      const paths = Object.entries(loadPaths(storedProject(stored))).flatMap(([load, project]) =>
        Object.entries(settlePaths(project)).map(([settle, state]) => [`${load}, ${settle}`, project, state] as const)
      );

      for (const [path, project, state] of paths) {
        expect(project.legacyLayout, `${name}, ${path}`).toBeDefined();
        expect(getActiveProject(state).legacyLayout, `${name}, ${path}`).toBeUndefined();
        // Opening a project is an entry, as it was then: the center fronts the stage. The others are no entry.
        expect(getLayoutOf(state).widgetRegions, `${name}, ${path}`).toEqual(
          path.endsWith(', open')
            ? {
                ...before.widgetRegions,
                center: {
                  ...before.widgetRegions.center,
                  activeInstanceId: getLayoutPreset(workspaceId).snapshot.widgetRegions.center.activeInstanceId,
                },
              }
            : before.widgetRegions
        );
        expect(state.notifications, `${name}, ${path}`).toEqual([]);
      }
    }
  });

  it('round-trips views through a save, the halves and a load, and drops junk ones without refusing', () => {
    const views = {
      compose: { collapsed: { right: true }, fronts: { left: 'upscale' }, sizes: { left: 500 } },
      edit: { windows: { preview: { mode: 'shaded' as const, x: 40 } } },
    };
    const project = { ...createInitialWorkbenchState().projects[0]!, views };
    const record = splitRecordOf(serializeProjectDocumentV3(project) as unknown as Record<string, unknown>, project);

    expect(record.workspace?.data).toMatchObject({ documentSchemaVersion: 5, views, workspaceId: 'compose' });
    expect(record.data).not.toHaveProperty('views');
    expect(loaded(deserializeProjectRecord(record)).views).toEqual(views);

    const junk = loaded(
      deserializeProjectRecord({
        ...record,
        workspace: {
          ...record.workspace!,
          data: {
            ...record.workspace!.data,
            views: {
              compose: { fronts: { nowhere: 'x' }, sizes: { left: 'wide' } },
              edit: 'junk',
              gone: views.compose,
            },
          },
        },
      })
    );

    expect(junk.views).toEqual({ gone: views.compose });
    // A workspace the account does not have loses its view once the editor knows the account.
    expect(getActiveProject(open(junk)).views).toEqual({});
  });

  it('starts a member with no workspace of their own on the new shape', () => {
    const project = createInitialWorkbenchState().projects[0]!;
    const record = splitRecordOf(serializeProjectDocumentV3(project) as unknown as Record<string, unknown>, project);
    const member = loaded(deserializeProjectRecord({ ...record, workspace: null }));

    expect(member.legacyLayout).toBeUndefined();
    expect(member).toMatchObject({ views: {}, workspaceId: 'compose' });
  });
});
