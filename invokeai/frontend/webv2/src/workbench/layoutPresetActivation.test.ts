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
    const activate = createLayoutPresetActivator({
      apply: (presetId) => appliedPresetIds.push(presetId),
      load: (preset) => (preset.id === 'compose' ? composeLoad.promise : editLoad.promise),
    });

    const composeActivation = activate(layoutPresets[0]);
    const editActivation = activate(layoutPresets[1]);

    editLoad.resolve();
    await editActivation;
    composeLoad.resolve();
    await composeActivation;

    expect(appliedPresetIds).toEqual(['edit']);
  });
});
