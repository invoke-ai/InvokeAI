import { getLayoutPreset } from '@workbench/layoutPresets';
import {
  addCustomWorkspace,
  createInitialWorkbenchState,
  getActiveProject,
  getLayoutOf,
  workbenchReducer,
} from '@workbench/workbenchState.testing';
import { describe, expect, it } from 'vitest';

import { getWorkspaceStructure } from './accountWorkspaces';
import { registryAvailability } from './layoutFixtures.testing';
import { createShippedStructureSelector } from './projectLayout';
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

  it('hides workspace switching while customizing', () => {
    const locked = { canEditStructure: false, isCustomizing: false };
    const customizing = { canEditStructure: true, isCustomizing: true };

    expect(isWorkspaceCommandAvailable('app.selectEditPreset', locked)).toBe(true);
    expect(isWorkspaceCommandAvailable('app.customizeWorkspace', locked)).toBe(true);
    expect(isWorkspaceCommandAvailable('app.toggleLeftPanel', locked)).toBe(true);

    expect(isWorkspaceCommandAvailable('app.toggleLeftPanel', customizing)).toBe(true);
    expect(isWorkspaceCommandAvailable('app.selectEditPreset', customizing)).toBe(false);
    expect(isWorkspaceCommandAvailable('app.backToPreviousWorkspace', customizing)).toBe(false);
    expect(isWorkspaceCommandAvailable('app.customizeWorkspace', customizing)).toBe(false);
  });
});

describe('Customize session', () => {
  const finish = { type: 'finishCustomize' } as const;
  const copy = { defaultRoute: null, iconId: 'star', id: 'review', label: 'Compose copy' };
  const sessionOf = (state: ReturnType<typeof onCompose>) =>
    getCustomizeSession(state.customizing, getActiveProject(state))!;
  const snapshotOf = (state: ReturnType<typeof onCompose>) => ({
    ...state,
    activeLayout: getLayoutOf(state),
    activeProject: getActiveProject(state),
  });
  // Fronted before Customize: a view of the workspace, which Cancel keeps and Done makes the default.
  const onQueue = () =>
    workbenchReducer(onCompose(), { region: 'right', type: 'selectRegionWidget', widgetId: 'queue' });

  it('edits a draft only; Cancel leaves the account and the view exactly as before Customize', () => {
    const before = onQueue();
    const customizing = workbenchReducer(before, { type: 'beginCustomize' });
    const removed = workbenchReducer(customizing, { region: 'right', type: 'toggleRegionWidget', widgetId: 'gallery' });
    const resized = workbenchReducer(removed, { region: 'left', sizePx: 600, type: 'setRegionWidgetSize' });

    expect(getLayoutOf(resized).widgetRegions.right.instanceIds).not.toContain('gallery');
    expect(resized.account).toBe(before.account);

    const cancelled = workbenchReducer(resized, { type: 'cancelCustomize' });

    expect(cancelled.account).toBe(before.account);
    expect(getActiveProject(cancelled).views).toEqual(getActiveProject(before).views);
    expect(getLayoutOf(cancelled)).toEqual(getLayoutOf(before));
    expect(getActiveProject(cancelled).undoRedo).toEqual({ future: [], past: [] });
    expect(getCustomizeSession(cancelled.customizing, getActiveProject(cancelled))).toBeNull();
  });

  it('Done writes the draft with what is on screen as its defaults in one step, and empties the view', () => {
    const customizing = workbenchReducer(onQueue(), { type: 'beginCustomize' });
    const removed = workbenchReducer(customizing, { region: 'right', type: 'toggleRegionWidget', widgetId: 'gallery' });
    const onScreen = getLayoutOf(removed);
    const done = workbenchReducer(removed, finish);
    const structure = getWorkspaceStructure(done.account, 'compose');

    expect(structure.widgetRegions.right).toMatchObject({
      activeInstanceId: 'queue',
      instanceIds: ['image-map', 'queue'],
    });
    expect(getActiveProject(done).views.compose).toBeUndefined();
    expect(getLayoutOf(done).widgetRegions).toEqual(onScreen.widgetRegions);
    expect(done.customizing).toEqual({});

    // Another project on Compose sees the new structure.
    const other = workbenchReducer(done, { type: 'createProject' });

    expect(getLayoutOf(other).widgetRegions.right.instanceIds).toEqual(['image-map', 'queue']);
  });

  it('writes no override for Done on an untouched built-in', () => {
    const customizing = workbenchReducer(onQueue(), { type: 'beginCustomize' });
    const done = workbenchReducer(customizing, finish);

    expect(done.account.layoutPresetOverrides?.compose).toBeUndefined();
    expect(getActiveProject(done).views).toEqual(getActiveProject(customizing).views);
  });

  it('steps its own Undo and Redo through every kind of edit in order, name and icon included', () => {
    const customizing = workbenchReducer(onCompose(), { type: 'beginCustomize' });
    const removed = workbenchReducer(customizing, { region: 'right', type: 'toggleRegionWidget', widgetId: 'gallery' });
    const named = workbenchReducer(workbenchReducer(removed, { draft: { label: 'Comp' }, type: 'setCustomizeDraft' }), {
      draft: { label: 'Composer' },
      type: 'setCustomizeDraft',
    });
    const iconed = workbenchReducer(named, { draft: { iconId: 'star' }, type: 'setCustomizeDraft' });
    const resized = workbenchReducer(iconed, { region: 'right', sizePx: 600, type: 'setRegionWidgetSize' });
    const undo = (state: typeof resized) => workbenchReducer(state, { type: 'undoProjectChange' });

    expect(sessionOf(undo(resized)).draft).toEqual({ iconId: 'star', label: 'Composer' });
    expect(getLayoutOf(undo(resized)).widgetRegions.right.sizePx).toBe(450);
    expect(sessionOf(undo(undo(resized))).draft).toEqual({ label: 'Composer' });
    // Typing the name was one step.
    expect(sessionOf(undo(undo(undo(resized)))).draft).toEqual({});
    expect(getLayoutOf(undo(undo(undo(resized)))).widgetRegions.right.instanceIds).not.toContain('gallery');
    expect(getLayoutOf(undo(undo(undo(undo(resized))))).widgetRegions.right.instanceIds).toContain('gallery');
    expect(sessionOf(workbenchReducer(undo(undo(resized)), { type: 'redoProjectChange' })).draft).toEqual({
      iconId: 'star',
      label: 'Composer',
    });
    expect(resized.account).toBe(customizing.account);
    expect(getActiveProject(resized).undoRedo.past).toHaveLength(0);
  });

  it('edits the route as a draft step, written at Done as one step Undo takes back', () => {
    const before = onQueue();
    const customizing = workbenchReducer(before, { type: 'beginCustomize' });
    const routed = workbenchReducer(customizing, {
      draft: { defaultRoute: { destination: 'canvas', sourceId: 'upscale' } },
      type: 'setCustomizeDraft',
    });
    const removed = workbenchReducer(routed, { region: 'right', type: 'toggleRegionWidget', widgetId: 'gallery' });

    expect(sessionOf(workbenchReducer(removed, { type: 'undoProjectChange' })).draft).toEqual(sessionOf(routed).draft);
    expect(sessionOf(workbenchReducer(routed, { type: 'undoProjectChange' })).draft).toEqual({});
    expect(routed.account).toBe(before.account);

    const done = workbenchReducer(removed, finish);
    const [step] = getActiveProject(done).undoRedo.past;

    expect(done.account.layoutPresetRouteOverrides?.compose).toEqual({ destination: 'canvas', sourceId: 'upscale' });
    expect(step?.notice).toEqual({ kind: 'updateWorkspace', presetId: 'compose' });

    const undone = workbenchReducer(done, { type: 'undoProjectChange' });

    expect(undone.account.layoutPresetRouteOverrides?.compose).toBeUndefined();
    expect(undone.account.layoutPresetOverrides?.compose).toBeUndefined();
    expect(getLayoutOf(undone).widgetRegions).toEqual(getLayoutOf(before).widgetRegions);
  });

  it('resets to default as a draft edit: undoable, removing the overrides at Done and none at Cancel', () => {
    const customized = workbenchReducer(
      workbenchReducer(
        workbenchReducer(workbenchReducer(onQueue(), { type: 'beginCustomize' }), {
          region: 'right',
          type: 'toggleRegionWidget',
          widgetId: 'gallery',
        }),
        { draft: { iconId: 'star', label: 'Mine' }, type: 'setCustomizeDraft' }
      ),
      finish
    );
    const customizing = workbenchReducer(customized, { type: 'beginCustomize' });
    const reset = workbenchReducer(customizing, { type: 'resetCustomizeToDefault' });
    const shipped = getLayoutPreset('compose').snapshot;

    expect(customized.account.layoutPresetOverrides?.compose).toBeDefined();
    expect(customized.account.layoutPresetMetadataOverrides?.compose).toEqual({ iconId: 'star', label: 'Mine' });
    expect(createShippedStructureSelector(registryAvailability)(snapshotOf(customizing))).toBe(false);
    expect(createShippedStructureSelector(registryAvailability)(snapshotOf(reset))).toBe(true);
    expect(getLayoutOf(reset).widgetRegions.right.instanceIds).toEqual(shipped.widgetRegions.right.instanceIds);
    expect(reset.account).toBe(customized.account);

    const undone = workbenchReducer(reset, { type: 'undoProjectChange' });

    expect(sessionOf(undone).resetPending).toBeUndefined();
    expect(getLayoutOf(undone).widgetRegions.right.instanceIds).not.toContain('gallery');
    expect(sessionOf(workbenchReducer(undone, { type: 'redoProjectChange' })).resetPending).toBe(true);
    expect(workbenchReducer(reset, { type: 'cancelCustomize' }).account).toBe(customized.account);

    const done = workbenchReducer(reset, finish);

    expect(done.account.layoutPresetOverrides?.compose).toBeUndefined();
    expect(done.account.layoutPresetMetadataOverrides?.compose).toBeUndefined();

    // Edited further after the reset, the edited result is what Done saves.
    const editedAfter = workbenchReducer(
      workbenchReducer(reset, { region: 'right', type: 'toggleRegionWidget', widgetId: 'queue' }),
      finish
    );

    expect(getWorkspaceStructure(editedAfter.account, 'compose').widgetRegions.right.instanceIds).toEqual([
      'gallery',
      'image-map',
    ]);
    expect(editedAfter.account.layoutPresetMetadataOverrides?.compose).toBeUndefined();
  });

  it('drafts a new workspace that only Done creates; Cancel leaves no trace', () => {
    const before = onQueue();
    const drafting = workbenchReducer(before, { newWorkspace: copy, type: 'beginCustomize' });
    const edited = workbenchReducer(drafting, { region: 'right', type: 'toggleRegionWidget', widgetId: 'gallery' });

    // Seen through an empty view of its own, it starts as what was on screen.
    expect(getLayoutOf(drafting).widgetRegions).toEqual(getLayoutOf(before).widgetRegions);
    expect(edited.account).toBe(before.account);
    expect(getActiveProject(edited).views.compose).toBe(getActiveProject(before).views.compose);

    const cancelled = workbenchReducer(edited, { type: 'cancelCustomize' });

    expect(cancelled.account).toBe(before.account);
    expect(getActiveProject(cancelled).views).toEqual(getActiveProject(before).views);
    expect(getActiveProject(cancelled).workspaceId).toBe('compose');
    expect(getLayoutOf(cancelled)).toEqual(getLayoutOf(before));

    const done = workbenchReducer(
      workbenchReducer(edited, { draft: { label: 'Review' }, type: 'setCustomizeDraft' }),
      finish
    );

    expect(done.account.customLayoutPresets).toEqual([
      expect.objectContaining({ iconId: 'star', id: 'review', label: 'Review' }),
    ]);
    expect(done.account.layoutPresetOrder?.at(-1)).toBe('review');
    expect(done.account.layoutPresetOverrides).toEqual(before.account.layoutPresetOverrides);
    expect(getActiveProject(done)).toMatchObject({ views: getActiveProject(before).views, workspaceId: 'review' });
    expect(getLayoutOf(done).widgetRegions).toEqual(getLayoutOf(edited).widgetRegions);
    expect(getLayoutOf(workbenchReducer(done, { presetId: 'compose', type: 'applyPreset' }))).toEqual(
      getLayoutOf(before)
    );
  });

  it('saves as new from inside Customize, leaving the original workspace and its view untouched', () => {
    const before = onQueue();
    const customizing = workbenchReducer(before, { type: 'beginCustomize' });
    const edited = workbenchReducer(
      workbenchReducer(customizing, { region: 'right', type: 'toggleRegionWidget', widgetId: 'gallery' }),
      { region: 'left', sizePx: 600, type: 'setRegionWidgetSize' }
    );
    const saving = workbenchReducer(edited, { type: 'saveCustomizeAsNew', workspace: copy });

    expect(getLayoutOf(saving).widgetRegions).toEqual(getLayoutOf(edited).widgetRegions);
    expect(getActiveProject(saving).views.compose).toBe(getActiveProject(before).views.compose);
    expect(saving.account).toBe(before.account);
    expect(workbenchReducer(saving, { type: 'cancelCustomize' }).account).toBe(before.account);
    expect(getActiveProject(workbenchReducer(saving, { type: 'cancelCustomize' })).views).toEqual(
      getActiveProject(before).views
    );

    const done = workbenchReducer(
      workbenchReducer(saving, { draft: { label: 'Review' }, type: 'setCustomizeDraft' }),
      finish
    );

    expect(done.account.layoutPresetOverrides?.compose).toBeUndefined();
    expect(getWorkspaceStructure(done.account, 'review').widgetRegions.right.instanceIds).not.toContain('gallery');
    expect(getActiveProject(done).views.compose).toBe(getActiveProject(before).views.compose);
    expect(getActiveProject(done).workspaceId).toBe('review');
  });

  it('ends when its workspace is deleted', () => {
    const custom = addCustomWorkspace(onCompose(), { id: 'mine', label: 'Mine' });
    const customizing = workbenchReducer(custom, { type: 'beginCustomize' });
    const deleted = workbenchReducer(customizing, { presetId: 'mine', type: 'deleteLayoutPreset' });

    expect(customizing.customizing).toMatchObject({ [custom.activeProjectId]: { draft: {}, presetId: 'mine' } });
    expect(deleted.customizing).toEqual({});

    // A new workspace drafted from it goes too, leaving no view of the draft.
    const drafting = workbenchReducer(workbenchReducer(custom, { newWorkspace: copy, type: 'beginCustomize' }), {
      region: 'left',
      sizePx: 600,
      type: 'setRegionWidgetSize',
    });
    const draftDeleted = workbenchReducer(drafting, { presetId: 'mine', type: 'deleteLayoutPreset' });

    expect(getActiveProject(drafting).views).toHaveProperty('review');
    expect(draftDeleted.customizing).toEqual({});
    expect(getActiveProject(draftDeleted).views).not.toHaveProperty('review');
  });
});
