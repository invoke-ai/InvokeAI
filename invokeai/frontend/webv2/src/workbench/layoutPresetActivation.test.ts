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
    await editActivation;
    composeLoad.resolve();
    await composeActivation;

    expect(appliedPresetIds).toEqual(['edit']);
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

    await activator.activate(layoutPresets[0]);

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
    await activation;

    expect(appliedPresetIds).toEqual([]);
  });
});
