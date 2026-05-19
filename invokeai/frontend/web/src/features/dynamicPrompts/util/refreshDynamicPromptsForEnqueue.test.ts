import type { RootState } from 'app/store/store';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  dynamicPromptsInitiate: vi.fn(),
  getShouldProcessPrompt: vi.fn((prompt: string) => prompt.includes('__')),
  selectPresetModifiedPrompts: vi.fn((state: RootState) => ({ positive: state.params.positivePrompt })),
}));

vi.mock('features/dynamicPrompts/util/getShouldProcessPrompt', () => ({
  getShouldProcessPrompt: mocks.getShouldProcessPrompt,
}));

vi.mock('features/nodes/util/graph/graphBuilderUtils', () => ({
  selectPresetModifiedPrompts: mocks.selectPresetModifiedPrompts,
}));

vi.mock('services/api/endpoints/utilities', () => ({
  utilitiesApi: {
    endpoints: {
      dynamicPrompts: {
        initiate: mocks.dynamicPromptsInitiate,
      },
    },
  },
}));

import { getDynamicPromptsOutputCount } from './resolveDynamicPrompts';
import {
  getShouldRefreshDynamicPromptsForEnqueue,
  refreshDynamicPromptsForEnqueue,
} from './refreshDynamicPromptsForEnqueue';

const buildState = (overrides: {
  mode?: 'random' | 'combinatorial';
  randomRefreshMode?: 'manual' | 'per_enqueue' | 'per_image';
  randomSamples?: number;
  iterations?: number;
  positivePrompt?: string;
}): RootState =>
  ({
    params: {
      iterations: overrides.iterations ?? 3,
      positivePrompt: overrides.positivePrompt ?? '__camera/lens__',
    },
    dynamicPrompts: {
      mode: overrides.mode ?? 'random',
      randomRefreshMode: overrides.randomRefreshMode ?? 'per_image',
      randomSamples: overrides.randomSamples ?? 1,
      maxCombinations: 100,
    },
  }) as RootState;

describe('refreshDynamicPromptsForEnqueue', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    mocks.dynamicPromptsInitiate.mockReset();
    mocks.getShouldProcessPrompt.mockClear();
    mocks.selectPresetModifiedPrompts.mockClear();
  });

  it('refreshes random prompts for per-image and per-invoke modes only', () => {
    expect(getShouldRefreshDynamicPromptsForEnqueue(buildState({ randomRefreshMode: 'per_image' }))).toBe(true);
    expect(getShouldRefreshDynamicPromptsForEnqueue(buildState({ randomRefreshMode: 'per_enqueue' }))).toBe(true);
    expect(getShouldRefreshDynamicPromptsForEnqueue(buildState({ randomRefreshMode: 'manual' }))).toBe(false);
    expect(getShouldRefreshDynamicPromptsForEnqueue(buildState({ mode: 'combinatorial' }))).toBe(false);
  });

  it('requests randomSamples times iterations for per-image enqueue refresh', () => {
    expect(
      getDynamicPromptsOutputCount({
        prompt: '__camera/lens__',
        randomRefreshMode: 'per_image',
        randomSamples: 2,
        iterations: 3,
      })
    ).toBe(6);
  });

  it('requests only randomSamples for per-invoke enqueue refresh', () => {
    expect(
      getDynamicPromptsOutputCount({
        prompt: '__camera/lens__',
        randomRefreshMode: 'per_enqueue',
        randomSamples: 2,
        iterations: 3,
      })
    ).toBe(2);
  });

  it('requests randomSamples times iterations for per-invoke cycle-only enqueue refresh', () => {
    expect(
      getDynamicPromptsOutputCount({
        prompt: '__@lighting/studio__',
        randomRefreshMode: 'per_enqueue',
        randomSamples: 2,
        iterations: 3,
      })
    ).toBe(6);
  });

  it('counts cycle-only locked-preview prompts per generated output', () => {
    expect(
      getDynamicPromptsOutputCount({
        prompt: '__@lighting/studio__',
        randomRefreshMode: 'manual',
        randomSamples: 2,
        iterations: 3,
      })
    ).toBe(6);
  });

  it('passes combinatorial false and a seed when refreshing per-image prompts', async () => {
    const request = { unwrap: vi.fn().mockResolvedValue({ prompts: ['35mm', '85mm', '50mm'], error: undefined }) };
    mocks.dynamicPromptsInitiate.mockReturnValue(request);
    vi.spyOn(Date, 'now').mockReturnValue(1000);
    vi.spyOn(Math, 'random').mockReturnValue(0);

    const state = buildState({ randomRefreshMode: 'per_image', randomSamples: 1, iterations: 3 });
    const dispatch = vi.fn((action) => action);
    const store = {
      dispatch,
      getState: vi.fn(() => state),
    };

    await refreshDynamicPromptsForEnqueue(store as never);

    expect(mocks.dynamicPromptsInitiate).toHaveBeenCalledWith(
      {
        prompt: '__camera/lens__',
        max_prompts: 3,
        combinatorial: false,
        seed: 1000,
      },
      { subscribe: false, forceRefetch: true }
    );
    expect(dispatch).toHaveBeenCalledWith(expect.objectContaining({ type: 'dynamicPrompts/promptsChanged' }));
  });

  it('resolves mixed cycle and random prompts with fixed random tokens for per-invoke enqueue refresh', async () => {
    const requests = [
      { unwrap: vi.fn().mockResolvedValue({ prompts: ['<<INVOKE_CYCLE_0>>, pearl earrings'], error: undefined }) },
      {
        unwrap: vi.fn().mockResolvedValue({
          prompts: ['35mm lens, pearl earrings', '50mm lens, pearl earrings', '85mm lens, pearl earrings'],
          error: undefined,
        }),
      },
    ];
    mocks.dynamicPromptsInitiate.mockReturnValueOnce(requests[0]).mockReturnValueOnce(requests[1]);
    vi.spyOn(Date, 'now').mockReturnValue(1000);
    vi.spyOn(Math, 'random').mockReturnValue(0);

    const state = buildState({
      randomRefreshMode: 'per_enqueue',
      iterations: 3,
      positivePrompt: '__@camera/lens__, __character/accessory__',
    });
    const dispatch = vi.fn((action) => action);
    const store = {
      dispatch,
      getState: vi.fn(() => state),
    };

    await refreshDynamicPromptsForEnqueue(store as never);

    expect(mocks.dynamicPromptsInitiate).toHaveBeenNthCalledWith(
      1,
      {
        prompt: '<<INVOKE_CYCLE_0>>, __character/accessory__',
        max_prompts: 1,
        combinatorial: false,
        seed: 1000,
      },
      { subscribe: false, forceRefetch: true }
    );
    expect(mocks.dynamicPromptsInitiate).toHaveBeenNthCalledWith(
      2,
      {
        prompt: '__@camera/lens__, pearl earrings',
        max_prompts: 3,
        combinatorial: false,
        seed: 1000,
      },
      { subscribe: false, forceRefetch: true }
    );
    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({
        payload: ['35mm lens, pearl earrings', '50mm lens, pearl earrings', '85mm lens, pearl earrings'],
        type: 'dynamicPrompts/promptsChanged',
      })
    );
  });
});
