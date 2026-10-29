import type { LayoutPreset } from '@workbench/layoutContracts';
import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { describe, expect, it, vi } from 'vitest';

import {
  doesProjectMatchLayoutPreset,
  findLayoutPresetWorkingCopy,
  getUnsavedInactiveLayoutPresetIds,
  resolveSavedLayoutPreset,
} from './layoutPresetSnapshots';
import { serializeProjectDocumentV3 } from './projects/projectDocument';
import { deserializeProjectDocument } from './projects/projectHydration';
import { normalizeWorkbenchProject, PRESET_WORKING_LAYOUT_LIMIT } from './workbenchState';
import { createInitialWorkbenchState, workbenchReducer, type WorkbenchAction } from './workbenchState.testing';
import { createWorkbenchStore } from './workbenchStore';

const WINDOW = { heightPx: 320, widthPx: 440, x: 210, y: 140 };

const getActiveProject = (state: WorkbenchState): Project =>
  state.projects.find((project) => project.id === state.activeProjectId)!;

const run = (state: WorkbenchState, ...actions: WorkbenchAction[]): WorkbenchState =>
  actions.reduce((next, action) => workbenchReducer(next, action), state);

const switchTo = (presetId: string): WorkbenchAction => ({ presetId, type: 'applyPreset' });
const addPreset = (presetId: string, label = presetId): WorkbenchAction => ({
  label,
  presetId,
  type: 'addLayoutPreset',
});

/** Rearrange the active preset: a resized rail and a floating window. */
const rearrange = (state: WorkbenchState, sizePx = 401): WorkbenchState =>
  run(
    state,
    { region: 'right', sizePx, type: 'setRegionWidgetSize' },
    { instanceId: 'gallery', type: 'floatWidget' },
    { instanceId: 'gallery', type: 'setFloatingWidgetGeometry', ...WINDOW }
  );

const copyOf = (state: WorkbenchState, presetId: string) =>
  findLayoutPresetWorkingCopy(getActiveProject(state).presetWorkingLayouts, presetId);
const copyIds = (state: WorkbenchState) =>
  (getActiveProject(state).presetWorkingLayouts ?? []).map(({ presetId }) => presetId);

const unsavedInactive = (state: WorkbenchState): string[] => {
  const project = getActiveProject(state);

  return getUnsavedInactiveLayoutPresetIds(project.presetWorkingLayouts, project.layout.presetId, state.account);
};

const isActiveDirty = (state: WorkbenchState): boolean => {
  const project = getActiveProject(state);

  return !doesProjectMatchLayoutPreset(project, resolveSavedLayoutPreset(state.account, project.layout.presetId));
};

/** Through the load boundary server records, recovery drafts and imports share, from JSON as stored. */
const loadDocument = (document: object): Project => {
  const result = deserializeProjectDocument(JSON.parse(JSON.stringify(document)) as Record<string, unknown>);

  if (result.status !== 'loaded') {
    throw new Error(`document did not load: ${result.status}`);
  }

  return result.project;
};

/** A reload: every open project's saved document back through loading, then workbench hydration with the account. */
const reload = (state: WorkbenchState): WorkbenchState =>
  run(createInitialWorkbenchState(), {
    state: {
      ...state,
      projects: state.projects.map((project) => loadDocument(serializeProjectDocumentV3(project))),
    },
    type: 'hydrateWorkbench',
  });

describe('layout preset working copies', () => {
  it('brings back a preset’s modified arrangement after A → B → A, floating window included', () => {
    let state = rearrange(run(createInitialWorkbenchState(), switchTo('compose')));

    state = run(state, switchTo('edit'));

    // Edit opens as saved: the rail size and the window are Compose's, not carried across.
    expect(getActiveProject(state).widgetRegions.right.sizePx).not.toBe(401);
    expect(getActiveProject(state).floatingWidgets?.gallery).toBeUndefined();
    expect(isActiveDirty(state)).toBe(false);
    expect(unsavedInactive(state)).toEqual(['compose']);

    state = run(state, switchTo('compose'));
    const project = getActiveProject(state);

    expect(project.layout.presetId).toBe('compose');
    expect(project.widgetRegions.right.sizePx).toBe(401);
    expect(project.floatingWidgets?.gallery).toMatchObject(WINDOW);
    // Compose is still unsaved, now as the live layout; no copy of it stays behind in the list.
    expect(isActiveDirty(state)).toBe(true);
    expect(project.presetWorkingLayouts).toBeUndefined();
  });

  it('keeps arrangement only: widget state stays on the project, never in a working copy', () => {
    let state = run(createInitialWorkbenchState(), switchTo('compose'), {
      type: 'patchGenerateSettings',
      values: { positivePrompt: 'a lighthouse at dusk' },
    });
    state = run(rearrange(state), switchTo('edit'));

    const copy = copyOf(state, 'compose');

    expect(copy).toBeDefined();
    expect(JSON.stringify(copy)).not.toContain('lighthouse');
    for (const instance of Object.values(copy!.widgetInstances)) {
      expect(Object.keys(instance).sort()).toEqual(['id', 'title', 'typeId']);
    }
  });

  it('switching away from an unmodified preset leaves no copy', () => {
    const state = run(createInitialWorkbenchState(), switchTo('compose'), switchTo('edit'));

    expect(getActiveProject(state).presetWorkingLayouts).toBeUndefined();
    expect(unsavedInactive(state)).toEqual([]);
  });

  it('holds several presets’ unsaved arrangements at once, each restored to its own', () => {
    let state = rearrange(run(createInitialWorkbenchState(), switchTo('compose')), 401);
    state = run(state, switchTo('edit'), { region: 'right', sizePx: 433, type: 'setRegionWidgetSize' });
    state = run(state, switchTo('video'));

    expect(new Set(unsavedInactive(state))).toEqual(new Set(['compose', 'edit']));

    expect(getActiveProject(run(state, switchTo('edit'))).widgetRegions.right.sizePx).toBe(433);
    expect(getActiveProject(run(state, switchTo('compose'))).widgetRegions.right.sizePx).toBe(401);
  });

  it('saves the active preset and an inactive working copy, after which neither reads as unsaved', () => {
    let state = rearrange(run(createInitialWorkbenchState(), switchTo('compose')), 401);
    state = run(state, switchTo('edit'), { region: 'right', sizePx: 433, type: 'setRegionWidgetSize' });

    // The inactive copy saves without switching to it. The copy stays (a lost account write would leave it
    // recoverable) but, equal to what was saved, no longer reads as unsaved.
    state = run(state, { presetId: 'compose', type: 'saveLayoutPreset' });
    expect(state.account.layoutPresetOverrides?.compose?.widgetRegions.right.sizePx).toBe(401);
    expect(state.account.layoutPresetOverrides?.compose?.floatingWidgets?.gallery).toMatchObject(WINDOW);
    expect(copyIds(state)).toEqual(['compose']);
    expect(unsavedInactive(state)).toEqual([]);
    expect(getActiveProject(state).widgetRegions.right.sizePx).toBe(433);

    // The active preset saves the live layout.
    expect(isActiveDirty(state)).toBe(true);
    state = run(state, { presetId: 'edit', type: 'saveLayoutPreset' });
    expect(isActiveDirty(state)).toBe(false);
    expect(state.account.layoutPresetOverrides?.edit?.widgetRegions.right.sizePx).toBe(433);

    // Switching to the saved one shows it as saved, and its copy leaves with the switch.
    state = run(state, switchTo('compose'));
    expect(isActiveDirty(state)).toBe(false);
    expect(getActiveProject(state).widgetRegions.right.sizePx).toBe(401);
    expect(copyIds(state)).toEqual([]);

    // A preset with nothing unsaved here has nothing to save.
    expect(run(state, { presetId: 'video', type: 'saveLayoutPreset' })).toBe(state);
  });

  it('drops another project’s stale copy of a preset saved elsewhere, so it can neither show nor overwrite the save', () => {
    // Project 1 leaves Compose rearranged and saves it from Edit; its own copy stays, equal to the save.
    let state = rearrange(run(createInitialWorkbenchState(), switchTo('compose')), 401);
    state = run(state, switchTo('edit'), { presetId: 'compose', type: 'saveLayoutPreset' });
    const firstProjectId = state.activeProjectId;
    expect(copyIds(state)).toEqual(['compose']);

    // Project 2 opens Compose as saved, changes it and saves it again.
    state = run(state, { type: 'createProject' }, switchTo('compose'));
    state = run(state, { region: 'right', sizePx: 433, type: 'setRegionWidgetSize' });
    state = run(state, { presetId: 'compose', type: 'saveLayoutPreset' });

    // Back in project 1, the copy equal to the old save is gone: no dot, and Compose opens as now saved.
    state = run(state, { projectId: firstProjectId, type: 'switchProject' });
    expect(copyIds(state)).toEqual([]);
    expect(unsavedInactive(state)).toEqual([]);
    expect(getActiveProject(run(state, switchTo('compose'))).widgetRegions.right.sizePx).toBe(433);
  });

  it('keeps another project’s own changes to a preset saved elsewhere', () => {
    let state = rearrange(run(createInitialWorkbenchState(), switchTo('compose')), 401);
    state = run(state, switchTo('edit'));
    const firstProjectId = state.activeProjectId;

    state = run(state, { type: 'createProject' }, switchTo('compose'));
    state = run(state, { region: 'right', sizePx: 433, type: 'setRegionWidgetSize' });
    state = run(state, { presetId: 'compose', type: 'saveLayoutPreset' });

    state = run(state, { projectId: firstProjectId, type: 'switchProject' });
    expect(unsavedInactive(state)).toEqual(['compose']);
    expect(getActiveProject(run(state, switchTo('compose'))).widgetRegions.right.sizePx).toBe(401);
  });

  it('does not read a copy that equals its saved preset as unsaved', () => {
    let state = rearrange(run(createInitialWorkbenchState(), switchTo('compose')), 401);
    state = run(state, switchTo('edit'));
    const copy = copyOf(state, 'compose')!;

    expect(unsavedInactive(state)).toEqual(['compose']);

    // Saved elsewhere (another project, another tab): the account now holds exactly this arrangement.
    state = { ...state, account: { ...state.account, layoutPresetOverrides: { compose: copy } } };

    expect(unsavedInactive(state)).toEqual([]);
  });

  it('reverts an inactive copy without switching, and the active preset to its saved layout, each undoably', () => {
    let state = rearrange(run(createInitialWorkbenchState(), switchTo('compose')), 401);
    state = run(state, switchTo('edit'), { region: 'right', sizePx: 433, type: 'setRegionWidgetSize' });

    state = run(state, { presetId: 'compose', type: 'revertLayoutPreset' });
    expect(unsavedInactive(state)).toEqual([]);
    expect(getActiveProject(state).widgetRegions.right.sizePx).toBe(433);
    expect(getActiveProject(run(state, switchTo('compose'))).floatingWidgets?.gallery).toBeUndefined();

    state = run(state, { type: 'undoProjectChange' });
    expect(unsavedInactive(state)).toEqual(['compose']);
    state = run(state, { type: 'redoProjectChange' });

    state = run(state, { presetId: 'edit', type: 'revertLayoutPreset' });
    expect(isActiveDirty(state)).toBe(false);
    state = run(state, { type: 'undoProjectChange' });
    expect(getActiveProject(state).widgetRegions.right.sizePx).toBe(433);

    // Choosing the active preset again (its shortcut) is a revert too, and undoable the same way.
    state = run(state, switchTo('edit'));
    expect(isActiveDirty(state)).toBe(false);
    state = run(state, { type: 'undoProjectChange' });
    expect(getActiveProject(state).widgetRegions.right.sizePx).toBe(433);
  });

  it('undoes a switch together with the copies it moved', () => {
    let state = rearrange(run(createInitialWorkbenchState(), switchTo('compose')));
    state = run(state, switchTo('edit'), { type: 'undoProjectChange' });

    expect(getActiveProject(state).layout.presetId).toBe('compose');
    expect(getActiveProject(state).widgetRegions.right.sizePx).toBe(401);
    expect(getActiveProject(state).presetWorkingLayouts).toBeUndefined();

    state = run(state, { type: 'redoProjectChange' });
    expect(unsavedInactive(state)).toEqual(['compose']);
  });

  it('keeps a renamed preset’s copy and drops a deleted preset’s from every open project', () => {
    let state = run(createInitialWorkbenchState(), {
      label: 'Review',
      presetId: 'custom-review',
      type: 'addLayoutPreset',
    });
    state = rearrange(run(state, switchTo('custom-review')), 401);
    state = run(state, switchTo('compose'), {
      label: 'Critique',
      presetId: 'custom-review',
      type: 'renameLayoutPreset',
    });
    expect(unsavedInactive(state)).toEqual(['custom-review']);

    const firstProjectId = state.activeProjectId;
    state = rearrange(run(state, { type: 'createProject' }, switchTo('custom-review')), 411);
    state = run(state, switchTo('compose'), { presetId: 'custom-review', type: 'deleteLayoutPreset' });

    for (const project of state.projects) {
      expect(findLayoutPresetWorkingCopy(project.presetWorkingLayouts, 'custom-review')).toBeUndefined();
    }
    expect(state.projects.find((project) => project.id === firstProjectId)?.presetWorkingLayouts).toBeUndefined();
  });

  it('moves a project off its deleted preset onto the default’s own working copy, keeping it', () => {
    let state = rearrange(run(createInitialWorkbenchState(), switchTo('compose')), 401);
    state = run(state, addPreset('custom-review', 'Review'), switchTo('compose'));
    // The project now has Compose clean and live; rearrange it again, then leave for the custom preset.
    state = run(state, { region: 'right', sizePx: 433, type: 'setRegionWidgetSize' }, switchTo('custom-review'));
    state = run(state, { region: 'right', sizePx: 461, type: 'setRegionWidgetSize' });
    expect(unsavedInactive(state)).toEqual(['compose']);

    state = run(state, { presetId: 'custom-review', type: 'deleteLayoutPreset' });
    const project = getActiveProject(state);

    // Compose comes back as this project left it; the deleted preset's arrangement went with the preset.
    expect(project.layout.presetId).toBe('compose');
    expect(project.widgetRegions.right.sizePx).toBe(433);
    expect(isActiveDirty(state)).toBe(true);
    expect(project.presetWorkingLayouts).toBeUndefined();
  });

  it('moves a project off its deleted preset onto the saved default when it has no copy of it', () => {
    let state = run(createInitialWorkbenchState(), switchTo('compose'), addPreset('custom-review'));
    state = run(state, switchTo('custom-review'), { region: 'right', sizePx: 461, type: 'setRegionWidgetSize' });

    state = run(state, { presetId: 'custom-review', type: 'deleteLayoutPreset' });

    expect(getActiveProject(state).layout.presetId).toBe('compose');
    expect(isActiveDirty(state)).toBe(false);
  });

  it('saves the arrangement as a new preset and moves the project onto it, leaving the old preset as saved', () => {
    let state = rearrange(run(createInitialWorkbenchState(), switchTo('compose')), 401);
    state = run(state, addPreset('custom-review', 'Review'));

    expect(getActiveProject(state).layout.presetId).toBe('custom-review');
    expect(isActiveDirty(state)).toBe(false);

    state = run(state, switchTo('compose'));
    expect(isActiveDirty(state)).toBe(false);
    expect(copyIds(state)).toEqual([]);
    state = run(state, switchTo('custom-review'));
    expect(getActiveProject(state).widgetRegions.right.sizePx).toBe(401);
    expect(getActiveProject(state).floatingWidgets?.gallery).toMatchObject(WINDOW);
  });

  it('drops copies of presets the account no longer has on the next switch', () => {
    let state = run(createInitialWorkbenchState(), addPreset('custom-review'), switchTo('compose'));
    state = rearrange(run(state, switchTo('custom-review')));
    state = run(state, switchTo('compose'));
    expect(copyIds(state)).toEqual(['custom-review']);
    // As another tab or session leaves it: the preset is gone from the account, its copy still on this project.
    state = { ...state, account: { ...state.account, customLayoutPresets: [] } };

    state = run(state, switchTo('edit'));

    expect(getActiveProject(state).presetWorkingLayouts).toBeUndefined();
  });

  it(`keeps at most ${String(PRESET_WORKING_LAYOUT_LIMIT)} copies, forgetting the oldest`, () => {
    const presetIds = Array.from({ length: PRESET_WORKING_LAYOUT_LIMIT + 2 }, (_, index) => `custom-${String(index)}`);
    let state = createInitialWorkbenchState();

    for (const presetId of presetIds) {
      state = run(state, addPreset(presetId));
    }
    for (const [index, presetId] of presetIds.entries()) {
      state = run(state, switchTo(presetId), { region: 'right', sizePx: 500 + index, type: 'setRegionWidgetSize' });
    }
    state = run(state, switchTo('compose'));

    const kept = copyIds(state);

    expect(kept).toHaveLength(PRESET_WORKING_LAYOUT_LIMIT);
    expect(kept).not.toContain('custom-0');
    expect(kept).not.toContain('custom-1');
    // Oldest first: the order they were left in.
    expect(kept).toEqual(presetIds.slice(2));
  });

  it('never shows one project’s working copies in another', () => {
    let state = rearrange(run(createInitialWorkbenchState(), switchTo('compose')), 401);
    state = run(state, switchTo('edit'));
    const firstProjectId = state.activeProjectId;

    state = run(state, { type: 'createProject' }, switchTo('edit'), switchTo('compose'));

    expect(getActiveProject(state).id).not.toBe(firstProjectId);
    expect(getActiveProject(state).widgetRegions.right.sizePx).not.toBe(401);
    expect(getActiveProject(state).floatingWidgets?.gallery).toBeUndefined();
    expect(unsavedInactive(state)).toEqual([]);
    expect(serializeProjectDocumentV3(getActiveProject(state))).not.toHaveProperty('presetWorkingLayouts');

    state = run(state, { projectId: firstProjectId, type: 'switchProject' });
    expect(unsavedInactive(state)).toEqual(['compose']);
  });
});

describe('layout preset working copies in the project document', () => {
  it('survive a save and reload, then restore on the next switch', () => {
    let state = rearrange(run(createInitialWorkbenchState(), switchTo('compose')), 401);
    state = run(state, switchTo('edit'));

    const reloaded = reload(state);

    expect(getActiveProject(reloaded).presetWorkingLayouts).toEqual(getActiveProject(state).presetWorkingLayouts);
    expect(unsavedInactive(reloaded)).toEqual(['compose']);

    const back = run(reloaded, switchTo('compose'));
    expect(getActiveProject(back).widgetRegions.right.sizePx).toBe(401);
    expect(getActiveProject(back).floatingWidgets?.gallery).toMatchObject(WINDOW);
  });

  it('drops a saved copy on reload once the account proves the save, and keeps it if the save was lost', () => {
    let state = rearrange(run(createInitialWorkbenchState(), switchTo('compose')), 401);
    state = run(state, switchTo('edit'));
    const accountBeforeSave = state.account;
    state = run(state, { presetId: 'compose', type: 'saveLayoutPreset' });
    expect(copyIds(state)).toEqual(['compose']);

    expect(copyIds(reload(state))).toEqual([]);

    // The account write never landed: the account that loads is the one from before the save.
    const lost = reload({ ...state, account: accountBeforeSave });
    expect(copyIds(lost)).toEqual(['compose']);
    expect(unsavedInactive(lost)).toEqual(['compose']);
  });

  it('drops a reloaded project’s copies of presets the account no longer has, so they hold no place', () => {
    let state = run(createInitialWorkbenchState(), addPreset('custom-review'), switchTo('compose'));
    state = rearrange(run(state, switchTo('custom-review')));
    state = rearrange(run(state, switchTo('compose')), 433);
    state = run(state, switchTo('edit'));
    expect(copyIds(state)).toEqual(['custom-review', 'compose']);

    // The preset was deleted while this project was closed: the account no longer has it.
    const reloaded = reload({ ...state, account: { ...state.account, customLayoutPresets: [] } });

    expect(copyIds(reloaded)).toEqual(['compose']);
  });

  it('loads a document written before working copies exactly as before, and writes it back without them', () => {
    const current = serializeProjectDocumentV3(getActiveProject(createInitialWorkbenchState()));
    const { presetWorkingLayouts: _absent, ...older } = current;

    expect(current).not.toHaveProperty('presetWorkingLayouts');

    const loaded = loadDocument(older);

    expect(loaded.presetWorkingLayouts).toBeUndefined();
    expect(serializeProjectDocumentV3(loaded)).toEqual(older);
  });

  it('keeps only well-formed copies of presets other than the active one', () => {
    let state = rearrange(run(createInitialWorkbenchState(), switchTo('compose')), 401);
    state = run(state, switchTo('edit'));
    const project = getActiveProject(state);
    const copy = copyOf(state, 'compose')!;

    const loaded = normalizeWorkbenchProject({
      ...project,
      presetWorkingLayouts: [
        // The active preset's copy is the live layout, so a stored one is stale.
        { presetId: 'edit', snapshot: copy },
        {
          presetId: 'video',
          snapshot: { ...copy, widgetRegions: { ...copy.widgetRegions, right: { activeInstanceId: 7 } } },
        },
        { presetId: 'automate', snapshot: 'not a layout' },
        { snapshot: copy },
        null,
        // A historical id names the current preset; the later of two copies of one preset wins.
        {
          presetId: 'compose',
          snapshot: {
            ...copy,
            widgetRegions: { ...copy.widgetRegions, right: { ...copy.widgetRegions.right, sizePx: 380 } },
          },
        },
        { presetId: 'gallery', snapshot: copy },
      ],
    } as unknown as Project);

    expect(loaded.presetWorkingLayouts?.map(({ presetId }) => presetId)).toEqual(['compose']);
    expect(loaded.presetWorkingLayouts?.[0]?.snapshot.layout.presetId).toBe('compose');
    expect(loaded.presetWorkingLayouts?.[0]?.snapshot.widgetRegions.right.sizePx).toBe(401);
    for (const malformed of [{}, { compose: copy }, 'copies']) {
      expect(
        normalizeWorkbenchProject({ ...project, presetWorkingLayouts: malformed } as unknown as Project)
          .presetWorkingLayouts
      ).toBeUndefined();
    }
  });
});

describe('switching to a preset through the store', () => {
  it('loads the widgets of the working copy it will lay out, not of the saved preset', async () => {
    const isLoaded = vi.fn((_preset: LayoutPreset) => false);
    const loadLayoutPresetWidgets = vi.fn((_preset: LayoutPreset) => Promise.resolve());
    const store = createWorkbenchStore(rearrange(run(createInitialWorkbenchState(), switchTo('compose'))), {
      isLoaded,
      loadLayoutPresetWidgets,
    });

    store.commands.layout.applyPreset('edit');
    const copy = findLayoutPresetWorkingCopy(store.getSnapshot().activeProject.presetWorkingLayouts, 'compose');
    expect(copy).toBeDefined();

    await store.commands.layout.activatePreset('compose');

    expect(isLoaded).toHaveBeenLastCalledWith(expect.objectContaining({ id: 'compose', snapshot: copy }));
    expect(loadLayoutPresetWidgets).toHaveBeenLastCalledWith(
      expect.objectContaining({ id: 'compose', snapshot: copy })
    );
    expect(store.getSnapshot().activeProject.floatingWidgets?.gallery).toMatchObject(WINDOW);
  });

  it('cancels a switch still loading when asked, leaving the active preset and its changes alone', async () => {
    let finishLoading!: () => void;
    const store = createWorkbenchStore(rearrange(run(createInitialWorkbenchState(), switchTo('compose'))), {
      isLoaded: () => false,
      loadLayoutPresetWidgets: () =>
        new Promise<void>((resolve) => {
          finishLoading = resolve;
        }),
    });

    const pending = store.commands.layout.activatePreset('edit');
    store.commands.layout.cancelPresetActivation();
    finishLoading();

    await expect(pending).resolves.toBeNull();
    expect(store.getSnapshot().activeProject.layout.presetId).toBe('compose');
    expect(store.getSnapshot().activeProject.widgetRegions.right.sizePx).toBe(401);
  });
});
