import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  apiFetchJson: vi.fn(),
  refetchClusterLabels: vi.fn(),
}));

vi.mock('@platform/transport/http', () => ({
  apiFetchJson: mocks.apiFetchJson,
}));

vi.mock('./imageMapStore', () => ({
  refetchClusterLabels: mocks.refetchClusterLabels,
}));

import { updateImageMapVocab } from './vocabulary';

const backendVocab = (state: string, terms: string[] = ['zebra']) => ({
  error: null,
  max_term_length: 64,
  max_terms: 500,
  state,
  terms,
});

/** Walk the watcher's backed-off poll schedule far enough to cover `polls` iterations. */
const advancePolls = async (polls: number): Promise<void> => {
  for (let i = 0; i < polls; i += 1) {
    // The interval doubles from 2s and caps at 30s; 30s covers every step.
    await vi.advanceTimersByTimeAsync(30_000);
  }
};

beforeEach(() => {
  vi.useFakeTimers();
  mocks.apiFetchJson.mockReset();
  mocks.refetchClusterLabels.mockReset();
});

afterEach(async () => {
  // Let any live watcher observe a terminal state so it cannot leak into the
  // next test through the module-level dedup flag.
  mocks.apiFetchJson.mockResolvedValue(backendVocab('ready'));
  await advancePolls(3);
  vi.useRealTimers();
});

describe('updateImageMapVocab rebuild watcher', () => {
  it('re-fetches map labels once the rebuild lands, even with no component mounted', async () => {
    mocks.apiFetchJson
      .mockResolvedValueOnce(backendVocab('building')) // the PUT
      .mockResolvedValueOnce(backendVocab('building')) // first poll
      .mockResolvedValueOnce(backendVocab('ready')); // second poll

    const vocab = await updateImageMapVocab(['zebra']);

    expect(vocab.state).toBe('building');
    expect(mocks.refetchClusterLabels).not.toHaveBeenCalled();

    await advancePolls(2);

    expect(mocks.refetchClusterLabels).toHaveBeenCalledTimes(1);
    // The polls hit the GET endpoint, not the PUT.
    expect(mocks.apiFetchJson.mock.calls[1]).toEqual(['/api/v1/image_map/vocab']);
  });

  it('does not re-fetch labels when the rebuild fails', async () => {
    mocks.apiFetchJson
      .mockResolvedValueOnce(backendVocab('building')) // the PUT
      .mockResolvedValueOnce({ ...backendVocab('error'), error: 'no text encoder' });

    await updateImageMapVocab(['zebra']);
    await advancePolls(1);

    expect(mocks.refetchClusterLabels).not.toHaveBeenCalled();
  });

  it('starts no watcher when the save needs no rebuild', async () => {
    mocks.apiFetchJson.mockResolvedValueOnce(backendVocab('unavailable'));

    await updateImageMapVocab(['zebra']);
    await advancePolls(1);

    // Only the PUT itself; no polling followed.
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);
    expect(mocks.refetchClusterLabels).not.toHaveBeenCalled();
  });

  it('coalesces overlapping saves into one watcher', async () => {
    mocks.apiFetchJson
      .mockResolvedValueOnce(backendVocab('building')) // PUT 1
      .mockResolvedValueOnce(backendVocab('building')) // PUT 2
      .mockResolvedValue(backendVocab('ready')); // every poll thereafter

    await updateImageMapVocab(['zebra']);
    await updateImageMapVocab(['zebra', 'okapi']);
    await advancePolls(2);

    // Two PUTs, one poll: the second save reused the live watcher, which acts
    // on the final state rather than on which save produced it.
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(3);
    expect(mocks.refetchClusterLabels).toHaveBeenCalledTimes(1);
  });
});
