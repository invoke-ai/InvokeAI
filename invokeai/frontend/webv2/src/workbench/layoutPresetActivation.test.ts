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
      load: () => new Promise(() => {}),
    });

    const activation = activator.activate(layoutPresets[0]);

    activator.invalidate();
    await expect(activation).resolves.toBeNull();

    expect(appliedPresetIds).toEqual([]);
  });
});
