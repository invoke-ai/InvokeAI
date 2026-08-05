import { describe, expect, it } from 'vitest';

import { mapWithConcurrency } from './concurrency';

const deferred = <T>(): { promise: Promise<T>; resolve: (value: T) => void } => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((settle) => {
    resolve = settle;
  });

  return { promise, resolve };
};

describe('mapWithConcurrency', () => {
  it('returns results in input order, not completion order', async () => {
    const results = await mapWithConcurrency([30, 20, 10], 3, async (delay) => {
      await new Promise<void>((resolve) => {
        setTimeout(resolve, delay);
      });

      return delay;
    });

    expect(results).toEqual([30, 20, 10]);
  });

  it('passes the index alongside each item', async () => {
    const results = await mapWithConcurrency(['a', 'b', 'c'], 2, (item, index) => Promise.resolve(`${index}:${item}`));

    expect(results).toEqual(['0:a', '1:b', '2:c']);
  });

  it('never runs more than `concurrency` mappers at once', async () => {
    const gates = Array.from({ length: 5 }, () => deferred<void>());
    let inFlight = 0;
    let peak = 0;

    const run = mapWithConcurrency(gates, 2, async (gate) => {
      inFlight += 1;
      peak = Math.max(peak, inFlight);
      await gate.promise;
      inFlight -= 1;
    });

    // Release one at a time so the pool must refill rather than burst.
    for (const gate of gates) {
      gate.resolve();
      await Promise.resolve();
    }

    await run;

    expect(peak).toBe(2);
  });

  it('starts no workers for an empty list', async () => {
    let calls = 0;

    const results = await mapWithConcurrency([], 4, () => {
      calls += 1;

      return Promise.resolve(null);
    });

    expect(results).toEqual([]);
    expect(calls).toBe(0);
  });

  it('rejects when a mapper rejects', async () => {
    await expect(
      mapWithConcurrency([1, 2], 2, (item) => (item === 2 ? Promise.reject(new Error('nope')) : Promise.resolve(item)))
    ).rejects.toThrow('nope');
  });
});
