import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { describe, expect, it } from 'vitest';

import { getProjectWidgetValues } from './widgetState';
import { createInitialWorkbenchState, workbenchReducer, type WorkbenchAction } from './workbenchState.testing';

const getActiveProject = (state: WorkbenchState): Project =>
  state.projects.find((project) => project.id === state.activeProjectId)!;

const run = (state: WorkbenchState, ...actions: WorkbenchAction[]): WorkbenchState =>
  actions.reduce((next, action) => workbenchReducer(next, action), state);

const topStep = (state: WorkbenchState) => getActiveProject(state).undoRedo.past.at(-1);

const isPlaced = (project: Project, instanceId: string): boolean =>
  Object.values(project.widgetRegions).some((region) => region.instanceIds.includes(instanceId)) ||
  project.floatingWidgets?.[instanceId] !== undefined;

describe('layout undo', () => {
  it('undoes a preset switch without taking back settings changed since', () => {
    let state = run(createInitialWorkbenchState(), { presetId: 'compose', type: 'applyPreset' });
    state = run(state, { presetId: 'edit', type: 'applyPreset' });

    expect(topStep(state)?.notice).toEqual({ kind: 'switchPreset', presetId: 'edit' });

    state = run(state, { type: 'patchWidgetValues', values: { positivePrompt: 'typed after' }, widgetId: 'generate' });
    state = run(state, { type: 'undoProjectChange' });

    expect(getActiveProject(state).layout.presetId).toBe('compose');
    expect(getProjectWidgetValues(getActiveProject(state), 'generate').positivePrompt).toBe('typed after');
  });

  it('makes removing a docked widget or closing a window one undoable step that names the widget', () => {
    let state = run(createInitialWorkbenchState(), { presetId: 'compose', type: 'applyPreset' });
    const region = (['left', 'right'] as const).find((candidate) =>
      getActiveProject(state).widgetRegions[candidate].instanceIds.includes('gallery')
    )!;

    state = run(state, { region, type: 'toggleRegionWidget', widgetId: 'gallery' });
    expect(topStep(state)?.notice).toEqual({ instanceId: 'gallery', kind: 'removeWidget' });
    expect(isPlaced(getActiveProject(state), 'gallery')).toBe(false);

    state = run(state, { type: 'undoProjectChange' });
    expect(getActiveProject(state).widgetRegions[region].instanceIds).toContain('gallery');

    state = run(
      state,
      { instanceId: 'gallery', type: 'floatWidget' },
      { instanceId: 'gallery', type: 'closeFloatingWidget' }
    );
    expect(topStep(state)?.notice).toEqual({ instanceId: 'gallery', kind: 'removeWidget' });

    state = run(state, { type: 'undoProjectChange' });
    expect(getActiveProject(state).floatingWidgets?.gallery).toBeDefined();
  });

  it('undoes back through a named step, taking later steps with it, and ignores a step no longer in the history', () => {
    let state = run(createInitialWorkbenchState(), { presetId: 'compose', type: 'applyPreset' });
    state = run(state, { presetId: 'edit', type: 'applyPreset' });
    const switchStep = topStep(state)!.id;
    // A route change is a step of its own, after the switch.
    state = run(state, { type: 'toggleRoutingLock' });

    state = run(state, { throughEntryId: switchStep, type: 'undoProjectChange' });
    expect(getActiveProject(state).layout.presetId).toBe('compose');
    expect(getActiveProject(state).invocation.sourceLocked).toBe(false);
    expect(getActiveProject(state).undoRedo.future).toHaveLength(2);

    const unchanged = run(state, { throughEntryId: switchStep, type: 'undoProjectChange' });
    expect(unchanged).toBe(state);
  });

  // Remove Gallery, open Upscale (a new instance) and set it up: undo takes back the open, not Upscale's settings.
  it('records opening a widget as its own step, and never loses an instance or its values', () => {
    let state = run(createInitialWorkbenchState(), { presetId: 'compose', type: 'applyPreset' });
    const region = (['left', 'right'] as const).find((candidate) =>
      getActiveProject(state).widgetRegions[candidate].instanceIds.includes('gallery')
    )!;

    state = run(state, { region, type: 'toggleRegionWidget', widgetId: 'gallery' });
    state = run(state, { createNew: true, region: 'left', type: 'openRegionWidget', widgetId: 'upscale' });
    const upscale = getActiveProject(state).widgetRegions.left.activeInstanceId;
    state = run(state, { instanceId: upscale, type: 'patchWidgetInstanceValues', values: { scale: 4 } });

    state = run(state, { type: 'undoProjectChange' });
    expect(isPlaced(getActiveProject(state), upscale)).toBe(false);
    expect(isPlaced(getActiveProject(state), 'gallery')).toBe(false);
    expect(getActiveProject(state).widgetInstances[upscale]?.state.values.scale).toBe(4);

    state = run(state, { type: 'undoProjectChange' });
    expect(isPlaced(getActiveProject(state), 'gallery')).toBe(true);
    expect(getActiveProject(state).widgetInstances[upscale]?.state.values.scale).toBe(4);
  });

  it('folds a dragged divider into one step', () => {
    let state = run(createInitialWorkbenchState(), { presetId: 'compose', type: 'applyPreset' });
    const before = getActiveProject(state).undoRedo.past.length;

    state = run(
      state,
      { region: 'right', sizePx: 400, type: 'setRegionWidgetSize' },
      { region: 'right', sizePx: 420, type: 'setRegionWidgetSize' },
      { region: 'right', sizePx: 440, type: 'setRegionWidgetSize' }
    );
    expect(getActiveProject(state).undoRedo.past).toHaveLength(before + 1);

    const resized = getActiveProject(state).widgetRegions.right.sizePx;
    state = run(state, { type: 'undoProjectChange' });
    expect(getActiveProject(state).widgetRegions.right.sizePx).not.toBe(resized);
  });

  it('acts on the project it names, not the active one', () => {
    let state = run(createInitialWorkbenchState(), { presetId: 'compose', type: 'applyPreset' });
    state = run(state, { presetId: 'edit', type: 'applyPreset' });
    const projectId = state.activeProjectId;
    state = { ...state, activeProjectId: 'elsewhere' };

    state = run(state, { projectId, type: 'undoProjectChange' });
    expect(state.projects.find((project) => project.id === projectId)?.layout.presetId).toBe('compose');
  });
});
