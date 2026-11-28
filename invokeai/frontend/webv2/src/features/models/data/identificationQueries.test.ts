import { describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({ bulkGetModelIdentification: vi.fn() }));

vi.mock('./api', () => ({
  bulkGetModelIdentification: api.bulkGetModelIdentification,
  getModelIdentification: vi.fn(),
}));

const collect = async <T>(source: AsyncIterable<T>): Promise<T[]> => {
  const items: T[] = [];

  for await (const item of source) {
    items.push(item);
  }

  return items;
};

describe('streamModelIdentifications', () => {
  const echo = (keys: string[]) => Promise.resolve(keys.map((key) => ({ error: 'x', key, status: 'failed' })));

  it('requests small batches in order and yields their outcomes', async () => {
    const { streamModelIdentifications } = await import('./identificationQueries');

    api.bulkGetModelIdentification.mockReset().mockImplementation(echo);
    const keys = Array.from({ length: 25 }, (_, index) => `k${index}`);

    const outcomes = await collect(streamModelIdentifications(keys, new AbortController().signal));

    expect(api.bulkGetModelIdentification.mock.calls.map(([batch]) => batch.length)).toEqual([10, 10, 5]);
    expect(outcomes.map((outcome) => outcome.key)).toEqual(keys);
  });

  it('issues no further batch once stopped, keeping what it has', async () => {
    const { streamModelIdentifications } = await import('./identificationQueries');
    const stop = new AbortController();

    api.bulkGetModelIdentification.mockReset().mockImplementation((keys: string[]) => {
      stop.abort();
      return echo(keys);
    });
    const keys = Array.from({ length: 25 }, (_, index) => `k${index}`);

    const outcomes = await collect(streamModelIdentifications(keys, new AbortController().signal, stop.signal));

    expect(api.bulkGetModelIdentification).toHaveBeenCalledTimes(1);
    expect(outcomes).toHaveLength(10);
  });
});
