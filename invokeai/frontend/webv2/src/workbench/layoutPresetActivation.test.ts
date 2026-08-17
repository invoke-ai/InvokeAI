import { describe, expect, it } from 'vitest';

import { createLayoutPresetActivator } from './layoutPresetActivation';
import { layoutPresets } from './layoutPresets';

const createDeferred = () => {
  let resolve!: () => void;
  const promise = new Promise<void>((next) => {
    resolve = next;
  });

  return { promise, resolve };
};

describe('layout preset activation', () => {
  it('applies only the most recently requested preset when loads finish out of order', async () => {
    const composeLoad = createDeferred();
    const editLoad = createDeferred();
    const appliedPresetIds: string[] = [];
    const activator = createLayoutPresetActivator({
      apply: (presetId) => appliedPresetIds.push(presetId),
      getActiveProjectId: () => 'project-a',
      isCurrent: () => true,
      isLoaded: () => false,
      load: (preset) => (preset.id === 'compose' ? composeLoad.promise : editLoad.promise),
    });

    const composeActivation = activator.activate(layoutPresets[0]);
    const editActivation = activator.activate(layoutPresets[1]);

    editLoad.resolve();
    await expect(editActivation).resolves.toBe('edit');
    composeLoad.resolve();
    await expect(composeActivation).resolves.toBeNull();

    expect(appliedPresetIds).toEqual(['edit']);
  });

  // Callers paint the selection before the store has it; a dropped activation
  // that resolved silently would strand the control on a preset nothing applied.
  it('reports the dropped activation when the active project changed under it', async () => {
    const appliedPresetIds: string[] = [];
    let activeProjectId = 'project-a';
    const activator = createLayoutPresetActivator({
      apply: (presetId) => appliedPresetIds.push(presetId),
      applyDeadlineMs: 0,
      getActiveProjectId: () => activeProjectId,
      isCurrent: () => true,
      isLoaded: () => false,
      load: () => new Promise(() => {}),
    });

    const activation = activator.activate(layoutPresets[0]);
    activeProjectId = 'project-b';

    await expect(activation).resolves.toBeNull();
    expect(appliedPresetIds).toEqual([]);
  });

  it('reports the dropped activation when the preset definition was replaced under it', async () => {
    const appliedPresetIds: string[] = [];
    const activator = createLayoutPresetActivator({
      apply: (presetId) => appliedPresetIds.push(presetId),
      applyDeadlineMs: 0,
      getActiveProjectId: () => 'project-a',
      isCurrent: () => false,
      isLoaded: () => false,
      load: () => new Promise(() => {}),
    });

    await expect(activator.activate(layoutPresets[0])).resolves.toBeNull();
    expect(appliedPresetIds).toEqual([]);
  });

  it('applies at the deadline when a widget implementation is still loading', async () => {
    const appliedPresetIds: string[] = [];
    const activator = createLayoutPresetActivator({
      apply: (presetId) => appliedPresetIds.push(presetId),
      applyDeadlineMs: 0,
      getActiveProjectId: () => 'project-a',
      isCurrent: () => true,
      isLoaded: () => false,
      load: () => new Promise(() => {}),
    });

    await expect(activator.activate(layoutPresets[0])).resolves.toBe(layoutPresets[0].id);

    expect(appliedPresetIds).toEqual([layoutPresets[0].id]);
  });

  it('discards a deadline apply when the request was superseded before it fired', async () => {
    const appliedPresetIds: string[] = [];
    const activator = createLayoutPresetActivator({
      apply: (presetId) => appliedPresetIds.push(presetId),
      applyDeadlineMs: 0,
      getActiveProjectId: () => 'project-a',
      isCurrent: () => true,
      isLoaded: () => false,
      load: () => new Promise(() => {}),
    });

    const activation = activator.activate(layoutPresets[0]);

    activator.invalidate();
    await expect(activation).resolves.toBeNull();

    expect(appliedPresetIds).toEqual([]);
  });

  it('applies synchronously when every widget is already loaded', () => {
    const appliedPresetIds: string[] = [];
    const activator = createLayoutPresetActivator({
      apply: (presetId) => appliedPresetIds.push(presetId),
      getActiveProjectId: () => 'project-a',
      isCurrent: () => true,
      isLoaded: () => true,
      load: () => Promise.resolve(),
    });

    // No await: a warm switch must commit in the caller's own task, not a
    // microtask later, so the click that triggered it paints with the layout.
    void activator.activate(layoutPresets[1]);

    expect(appliedPresetIds).toEqual(['edit']);
  });

  it('still honours the deadline when a widget is cold', async () => {
    const appliedPresetIds: string[] = [];
    const activator = createLayoutPresetActivator({
      apply: (presetId) => appliedPresetIds.push(presetId),
      applyDeadlineMs: 0,
      getActiveProjectId: () => 'project-a',
      isCurrent: () => true,
      isLoaded: () => false,
      load: () => new Promise(() => {}),
    });

    const activation = activator.activate(layoutPresets[0]);

    expect(appliedPresetIds).toEqual([]);
    await activation;
    expect(appliedPresetIds).toEqual([layoutPresets[0].id]);
  });

  // Condition 2 of the fast-path contract: the fast path must bump
  // latestRequestId itself, not rely on apply's downstream dispatch to do it.
  // If it didn't, an earlier slow activation — still racing its deadline —
  // would see requestId === latestRequestId when its timer fires and would
  // overwrite the fast-path's result.
  it('bumps the request id so an earlier pending slow activation cannot overwrite a later fast-path apply', async () => {
    const appliedPresetIds: string[] = [];
    let loaded = false;
    const activator = createLayoutPresetActivator({
      apply: (presetId) => appliedPresetIds.push(presetId),
      applyDeadlineMs: 0,
      getActiveProjectId: () => 'project-a',
      isCurrent: () => true,
      isLoaded: () => loaded,
      load: () => new Promise(() => {}),
    });

    // Start a slow activation for a cold preset; it will sit waiting for its
    // deadline timer.
    const slowActivation = activator.activate(layoutPresets[0]);

    // A later, fully-warm activation for a different preset takes the fast
    // path and applies synchronously.
    loaded = true;
    void activator.activate(layoutPresets[1]);
    expect(appliedPresetIds).toEqual(['edit']);

    // When the slow activation's deadline fires, it must see it has been
    // superseded and must not clobber the fast-path result.
    await expect(slowActivation).resolves.toBeNull();
    expect(appliedPresetIds).toEqual(['edit']);
  });
});
