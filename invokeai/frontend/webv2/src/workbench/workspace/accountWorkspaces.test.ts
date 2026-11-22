import { getLayoutPreset } from '@workbench/layoutPresets';
import { createInitialWorkbenchState } from '@workbench/workbenchState.testing';
import { describe, expect, it } from 'vitest';

import {
  getWorkspaceStructure,
  readWorkspaceRecord,
  renameWorkspace,
  setWorkspaceRoute,
  writeWorkspaceRecord,
  writeWorkspaceStructure,
} from './accountWorkspaces';
import { registryAvailability } from './layoutFixtures.testing';
import { normalizeStructure } from './structure';
import { reduceStructure } from './structureReducer';

const account = createInitialWorkbenchState().account;
const shippedCompose = normalizeStructure(getLayoutPreset('compose').snapshot);
const withoutQueue = reduceStructure(
  shippedCompose,
  { instanceId: 'queue', region: 'right', type: 'removeWidget' },
  registryAvailability
);

describe('account workspaces', () => {
  it('keeps a built-in following what it ships with: a structure equal to the shipped one is no override', () => {
    const customized = writeWorkspaceStructure(account, 'compose', withoutQueue);

    expect(customized.layoutPresetOverrides?.compose).toBeDefined();
    expect(writeWorkspaceStructure(customized, 'compose', shippedCompose).layoutPresetOverrides).toEqual({});
    expect(writeWorkspaceStructure(account, 'compose', shippedCompose)).toBe(account);
  });

  it('writes a built-in’s name and route only where they differ from what it ships with', () => {
    const renamed = renameWorkspace(account, 'compose', 'Writing');
    const routed = setWorkspaceRoute(renamed, 'compose', { destination: 'canvas', sourceId: 'upscale' });

    expect(routed.layoutPresetMetadataOverrides?.compose).toEqual({ label: 'Writing' });
    expect(routed.layoutPresetRouteOverrides?.compose).toEqual({ destination: 'canvas', sourceId: 'upscale' });
    expect(renameWorkspace(routed, 'compose', 'Compose').layoutPresetMetadataOverrides?.compose).toBeUndefined();
    expect(setWorkspaceRoute(routed, 'compose', null).layoutPresetRouteOverrides?.compose).toBeUndefined();
  });

  it('puts a workspace’s record back exactly, never recreating a deleted custom workspace', () => {
    const before = readWorkspaceRecord(account, 'compose');
    const customized = renameWorkspace(writeWorkspaceStructure(account, 'compose', withoutQueue), 'compose', 'Writing');
    const restored = writeWorkspaceRecord(customized, 'compose', before);

    expect(readWorkspaceRecord(restored, 'compose')).toEqual(before);
    expect(getWorkspaceStructure(restored, 'compose')).toBe(getLayoutPreset('compose').snapshot);
    expect(
      writeWorkspaceRecord(account, 'gone', {
        kind: 'custom',
        preset: { id: 'gone', label: 'Gone', snapshot: shippedCompose },
      })
    ).toBe(account);
  });
});
