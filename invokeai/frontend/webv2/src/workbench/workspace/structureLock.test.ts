import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState.testing';
import { describe, expect, it } from 'vitest';

import { getCustomizeSession, isWorkspaceCommandAvailable, selectStructureLock } from './structureLock';

const onCompose = () => workbenchReducer(createInitialWorkbenchState(), { presetId: 'compose', type: 'applyPreset' });
const lockOf = (state: ReturnType<typeof onCompose>, rearrangeWithoutCustomize = false) =>
  selectStructureLock(
    { activeProject: state.projects.find(({ id }) => id === state.activeProjectId)!, customizing: state.customizing },
    rearrangeWithoutCustomize
  );

describe('structure lock', () => {
  it('opens structure while customizing this project on this workspace, or everywhere when rearranging is on', () => {
    const locked = onCompose();

    expect(lockOf(locked)).toEqual({ canEditStructure: false, isCustomizing: false });
    expect(lockOf(locked, true)).toEqual({ canEditStructure: true, isCustomizing: false });

    const customizing = workbenchReducer(locked, { type: 'beginCustomize' });

    expect(lockOf(customizing)).toEqual({ canEditStructure: true, isCustomizing: true });

    const otherProject = workbenchReducer(customizing, { type: 'createProject' });

    expect(lockOf(otherProject)).toEqual({ canEditStructure: false, isCustomizing: false });
    // Returning to the project shows its Customize again.
    expect(
      lockOf(workbenchReducer(otherProject, { projectId: locked.activeProjectId, type: 'switchProject' }))
    ).toEqual({
      canEditStructure: true,
      isCustomizing: true,
    });
  });

  it('hides structural commands while locked and workspace switching while customizing', () => {
    const locked = { canEditStructure: false, isCustomizing: false };
    const customizing = { canEditStructure: true, isCustomizing: true };

    expect(isWorkspaceCommandAvailable('app.saveLayoutPreset', locked)).toBe(false);
    expect(isWorkspaceCommandAvailable('app.resetPanelLayout', locked)).toBe(false);
    expect(isWorkspaceCommandAvailable('app.selectEditPreset', locked)).toBe(true);
    expect(isWorkspaceCommandAvailable('app.customizeWorkspace', locked)).toBe(true);
    expect(isWorkspaceCommandAvailable('app.toggleLeftPanel', locked)).toBe(true);

    expect(isWorkspaceCommandAvailable('app.saveLayoutPreset', customizing)).toBe(true);
    expect(isWorkspaceCommandAvailable('app.selectEditPreset', customizing)).toBe(false);
    expect(isWorkspaceCommandAvailable('app.backToPreviousWorkspace', customizing)).toBe(false);
    expect(isWorkspaceCommandAvailable('app.customizeWorkspace', customizing)).toBe(false);
  });
});

describe('Customize session', () => {
  it('starts and ends on a fresh history; Cancel lays out what was on screen at entry, unsaved changes included', () => {
    // Removed before Customize, as an account that rearranges without it may: Cancel must not bring it back.
    const unsaved = workbenchReducer(onCompose(), { region: 'right', type: 'toggleRegionWidget', widgetId: 'queue' });
    const customizing = workbenchReducer(unsaved, { type: 'beginCustomize' });
    const projectOf = (state: typeof unsaved) => state.projects.find(({ id }) => id === state.activeProjectId)!;

    expect(projectOf(unsaved).undoRedo.past).toHaveLength(1);
    expect(projectOf(customizing).undoRedo.past).toHaveLength(0);

    const removed = workbenchReducer(customizing, { region: 'right', type: 'toggleRegionWidget', widgetId: 'gallery' });
    const cancelled = workbenchReducer(removed, { revert: true, type: 'endCustomize' });

    expect(projectOf(removed).widgetRegions.right.instanceIds).not.toContain('gallery');
    expect(projectOf(cancelled).widgetRegions.right.instanceIds).toContain('gallery');
    expect(projectOf(cancelled).widgetRegions.right.instanceIds).not.toContain('queue');
    expect(projectOf(cancelled).undoRedo).toEqual({ future: [], past: [] });
    expect(getCustomizeSession(cancelled.customizing, projectOf(cancelled))).toBeNull();
  });

  it('ends when its workspace is deleted or the project moves onto a copy, where it can go on', () => {
    const custom = workbenchReducer(onCompose(), { label: 'Mine', presetId: 'mine', type: 'addLayoutPreset' });
    const customizing = workbenchReducer(custom, { type: 'beginCustomize' });
    const deleted = workbenchReducer(customizing, { presetId: 'mine', type: 'deleteLayoutPreset' });

    expect(customizing.customizing).toMatchObject({ [custom.activeProjectId]: { draft: {}, presetId: 'mine' } });
    expect(deleted.customizing).toEqual({});

    const copied = workbenchReducer(customizing, { label: 'Copy', presetId: 'copy', type: 'addLayoutPreset' });

    expect(copied.customizing).toEqual({});
    expect(workbenchReducer(copied, { type: 'beginCustomize' }).customizing).toMatchObject({
      [custom.activeProjectId]: { draft: {}, presetId: 'copy' },
    });
  });
});
