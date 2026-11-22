import type { AccountState, Project, WorkbenchState } from '@workbench/projectContracts';
import type { WidgetInstanceContract } from '@workbench/widgetContracts';

import {
  createInitialWorkbenchState,
  getActiveProject,
  getLayoutOf,
  workbenchReducer,
} from '@workbench/workbenchState.testing';
import { getWorkspaceStructure } from '@workbench/workspace/accountWorkspaces';
import { describe, expect, it } from 'vitest';

import type { ProjectWithWorkspaceDTO } from './api';

// Captured once from commit 7b783591bc, the last build before the split: a throwaway test there drove its reducer
// (`workbenchState.testing`) through each scenario's gestures, serialized the active project with its
// `serializeProjectDocumentV3`, and kept the layout fields, each instance's type and the account's workspaces.
import captured from './preSplitLayouts.fixtures.json';
import { serializeProjectDocumentV3 } from './projectDocument';
import { splitProjectDocument } from './projectHalves';
import { deserializeProjectRecord } from './projectHydration';

/** What the build before the split stored for one project, and the account beside it (`preSplitLayouts.fixtures`). */
type Captured = (typeof captured)[keyof typeof captured];

const base = createInitialWorkbenchState().projects[0]!;

/**
 * The server record of a project the build before the split saved: this build's content, and a workspace half holding
 * the captured layout fields at stamp 4, as that build wrote them.
 */
const recordOf = (
  id: string,
  { document }: Captured,
  updatedAt: string,
  edit: (stored: Record<string, unknown>) => Record<string, unknown> = (stored) => stored
): ProjectWithWorkspaceDTO => {
  const { views: _views, workspaceId: _workspaceId, ...current } = serializeProjectDocumentV3({ ...base, id });
  const instanceOf = (instanceId: string, typeId: string): WidgetInstanceContract =>
    base.widgetInstances[instanceId] ?? {
      ...Object.values(base.widgetInstances).find((instance) => instance.typeId === typeId)!,
      id: instanceId,
    };
  const { widgetInstances, ...layout } = document;
  const halves = splitProjectDocument({
    ...current,
    ...edit(layout),
    widgetInstances: Object.fromEntries(
      Object.entries(widgetInstances).map(([instanceId, typeId]) => [instanceId, instanceOf(instanceId, typeId)])
    ),
  } as unknown as ReturnType<typeof serializeProjectDocumentV3>);

  return {
    board_id: 'board',
    created_at: '2026-01-01T00:00:00Z',
    data: halves.content as unknown as Record<string, unknown>,
    minimum_canvas_schema_version: 3,
    minimum_document_schema_version: 4,
    name: id,
    project_id: id,
    revision: 1,
    updated_at: '2026-01-01T00:00:00Z',
    workspace: { data: halves.workspace as unknown as Record<string, unknown>, revision: 1, updated_at: updatedAt },
    workspace_updated_at: updatedAt,
  };
};

const load = (record: ProjectWithWorkspaceDTO): Project => {
  const result = deserializeProjectRecord(record);

  if (result.status !== 'loaded') {
    throw new Error(`Expected the record to load, got ${result.status}.`);
  }

  return result.project;
};

/** The first editor load after the upgrade, of an account that stored these projects; the first opens active. */
const upgrade = (records: readonly ProjectWithWorkspaceDTO[], account: Partial<AccountState> = {}): WorkbenchState => {
  const projects = records.map(load);

  return workbenchReducer(createInitialWorkbenchState(), {
    state: {
      ...createInitialWorkbenchState(),
      account: { activeLayoutPresetId: 'compose', ...account },
      activeProjectId: projects[0]!.id,
      projects,
    },
    type: 'hydrateWorkbench',
  });
};

const regionsOf = (state: WorkbenchState) => getLayoutOf(state).widgetRegions;

describe('projects saved by the build before the split', () => {
  it('reads when each was saved from its record, so the most recently used one adopts whatever the order', () => {
    const older = recordOf('older', captured['removed a widget'], '2026-01-01T00:00:00Z');
    const newer = recordOf('newer', captured['moved a tab to another rail'], '2026-03-01T00:00:00Z');
    const fresh = { ...base, id: 'fresh' };

    expect(load(newer).legacyLayout?.storedAt).toBe('2026-03-01T00:00:00Z');

    for (const records of [
      [older, newer],
      [newer, older],
    ]) {
      const state = workbenchReducer(createInitialWorkbenchState(), {
        state: {
          ...createInitialWorkbenchState(),
          account: { activeLayoutPresetId: 'compose' },
          activeProjectId: fresh.id,
          projects: [fresh, ...records.map(load)],
        },
        type: 'hydrateWorkbench',
      });

      expect(getWorkspaceStructure(state.account, 'compose').widgetRegions.left.instanceIds).toContain('image-map');
      expect(getWorkspaceStructure(state.account, 'compose').widgetRegions.right.instanceIds).toContain('queue');
    }
  });

  it('shows the active project as it looked, its arrangements adopted as the workspaces', () => {
    const looked = upgrade([recordOf('p', captured['changed only how it looked'], '2026-01-01T00:00:00Z')]);

    expect(looked.account.layoutPresetOverrides).toEqual({});
    expect(regionsOf(looked).right).toMatchObject({ activeInstanceId: 'queue', isCollapsed: true });
    expect(regionsOf(looked).left.sizePx).toBe(600);

    const floated = upgrade([recordOf('p', captured['floated a widget'], '2026-01-01T00:00:00Z')]);

    expect(Object.keys(getLayoutOf(floated).floatingWidgets ?? {})).toEqual(['image-map']);
    expect(regionsOf(floated).right.instanceIds).not.toContain('image-map');

    expect(
      regionsOf(upgrade([recordOf('p', captured['moved a tab to another rail'], '2026-01-01')])).left.instanceIds
    ).toContain('image-map');
    expect(regionsOf(upgrade([recordOf('p', captured['removed a widget'], '2026-01-01')])).right.instanceIds).toEqual([
      'gallery',
      'image-map',
    ]);
  });

  it('adopts a rearranged inactive workspace, a custom one, and a legacy preset id as their workspaces', () => {
    const inactive = upgrade([recordOf('p', captured['rearranged an inactive workspace'], '2026-01-01')]);

    expect(getActiveProject(inactive).workspaceId).toBe('compose');
    expect(getWorkspaceStructure(inactive.account, 'edit').widgetRegions.right.instanceIds).not.toContain('preview');

    const legacyId = upgrade([
      recordOf('p', captured['rearranged an inactive workspace'], '2026-01-01', (stored) => ({
        ...stored,
        presetWorkingLayouts: (stored.presetWorkingLayouts as { presetId: string }[]).map((copy) => ({
          ...copy,
          presetId: 'canvas',
        })),
      })),
    ]);

    expect(getWorkspaceStructure(legacyId.account, 'edit').widgetRegions.right.instanceIds).not.toContain('preview');

    const custom = captured['on a custom workspace'];
    const onCustom = upgrade([recordOf('p', custom, '2026-01-01')], custom.account as Partial<AccountState>);

    expect(getActiveProject(onCustom).workspaceId).toBe('custom-desk');
    expect(getWorkspaceStructure(onCustom.account, 'custom-desk').widgetRegions.right.instanceIds).not.toContain(
      'image-map'
    );
    expect(onCustom.account.layoutPresetOverrides).toEqual({});
  });

  it('lets the live layout outrank a copy of the preset it is on', () => {
    const moved = captured['moved a tab to another rail'];
    const state = upgrade([
      recordOf('p', moved, '2026-01-01', (stored) => ({
        ...stored,
        presetWorkingLayouts: [
          { presetId: 'compose', snapshot: { ...captured['removed a widget'].document, widgetInstances: {} } },
        ],
      })),
    ]);

    expect(regionsOf(state).left.instanceIds).toContain('image-map');
    expect(regionsOf(state).right.instanceIds).toContain('queue');
  });

  it('gives every project on an adopted workspace values for an instance the adopted arrangement created', () => {
    const added = captured['added a second Upscale'];
    const minted = Object.keys(added.document.widgetInstances).find((id) => id.startsWith('widget-upscale-'))!;
    const state = upgrade([recordOf('p', added, '2026-01-01')]);
    const other = workbenchReducer(state, { type: 'createProject' });

    expect(regionsOf(other).left.instanceIds).toContain(minted);
    expect(getActiveProject(other).widgetInstances[minted]?.typeId).toBe('upscale');
  });
});
