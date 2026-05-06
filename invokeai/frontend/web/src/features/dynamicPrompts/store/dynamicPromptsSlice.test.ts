import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import { describe, expect, it } from 'vitest';
import type { z } from 'zod';

import { dynamicPromptsSliceConfig, modeChanged, randomRefreshModeChanged } from './dynamicPromptsSlice';

describe('dynamicPromptsSlice', () => {
  type InitialState = ReturnType<typeof dynamicPromptsSliceConfig.getInitialState>;
  type SchemaState = z.infer<typeof dynamicPromptsSliceConfig.schema>;

  const { reducer } = dynamicPromptsSliceConfig.slice;
  const migrate = dynamicPromptsSliceConfig.persistConfig?.migrate;

  it('keeps the initial state aligned with the persisted schema', () => {
    assert<Equals<InitialState, SchemaState>>();
  });

  it('defaults fresh users to random mode', () => {
    expect(dynamicPromptsSliceConfig.getInitialState()).toMatchObject({
      _version: 3,
      mode: 'random',
      randomSamples: 1,
      maxCombinations: 100,
      randomSeed: 0,
      randomRefreshMode: 'per_enqueue',
    });
  });

  it('can switch to all-combinations mode', () => {
    const state = reducer(dynamicPromptsSliceConfig.getInitialState(), modeChanged('combinatorial'));

    expect(state.mode).toBe('combinatorial');
  });

  it('can switch random refresh behavior', () => {
    const state = reducer(dynamicPromptsSliceConfig.getInitialState(), randomRefreshModeChanged('manual'));

    expect(state.randomRefreshMode).toBe('manual');
  });

  it('migrates existing combinatorial users to all-combinations mode', () => {
    expect(migrate).toBeDefined();

    const state = migrate?.({
      _version: 1,
      maxPrompts: 250,
      combinatorial: true,
      seedBehaviour: 'PER_PROMPT',
    });

    expect(state).toMatchObject({
      _version: 3,
      mode: 'combinatorial',
      maxCombinations: 250,
      randomSamples: 1,
      randomRefreshMode: 'manual',
      seedBehaviour: 'PER_PROMPT',
    });
  });

  it('migrates existing random users to random mode', () => {
    expect(migrate).toBeDefined();

    const state = migrate?.({
      _version: 1,
      maxPrompts: 12,
      combinatorial: false,
    });

    expect(state).toMatchObject({
      _version: 3,
      mode: 'random',
      maxCombinations: 12,
      randomSamples: 1,
      randomRefreshMode: 'per_enqueue',
    });
  });

  it('migrates version 2 random users to reroll on invoke', () => {
    expect(migrate).toBeDefined();

    const state = migrate?.({
      _version: 2,
      mode: 'random',
      randomSamples: 1,
      maxCombinations: 100,
      randomSeed: 123,
      prompts: ['test'],
      isError: false,
      isLoading: false,
      seedBehaviour: 'PER_ITERATION',
    });

    expect(state).toMatchObject({
      _version: 3,
      mode: 'random',
      randomRefreshMode: 'per_enqueue',
    });
  });
});
