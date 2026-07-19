import { describe, expect, it, vi } from 'vitest';

import { createSingleFlight } from './singleFlight';

const deferred = <T>(): { promise: Promise<T>; resolve: (value: T) => void } => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });

  return { promise, resolve };
};

describe('platform createSingleFlight', () => {
  it('shares one in-flight promise between concurrent same-key callers', async () => {
    const flight = createSingleFlight<string>();
    const { promise, resolve } = deferred<string>();
    const task = vi.fn(() => promise);

    const first = flight.run('a', task);
    const second = flight.run('a', task);

    expect(second).toBe(first);
    expect(task).toHaveBeenCalledTimes(1);
    resolve('done');
    await expect(first).resolves.toBe('done');
  });

  it('starts a fresh task after the previous flight settles', async () => {
    const flight = createSingleFlight<number>();
    const task = vi.fn(() => Promise.resolve(1));

    await flight.run('a', task);
    await flight.run('a', task);

    expect(task).toHaveBeenCalledTimes(2);
  });

  it('starts a fresh task for a different key while one is in flight', async () => {
    const flight = createSingleFlight<string>();
    const flightA = deferred<string>();
    const taskA = vi.fn(() => flightA.promise);
    const taskB = vi.fn(() => Promise.resolve('b'));

    const a = flight.run('a', taskA);
    const b = flight.run('b', taskB);

    expect(b).not.toBe(a);
    expect(taskB).toHaveBeenCalledTimes(1);
    flightA.resolve('a');
    await Promise.all([a, b]);
  });

  it('does not let a superseded flight clear a newer one when it settles', async () => {
    const flight = createSingleFlight<string>();
    const oldFlight = deferred<string>();
    const newFlight = deferred<string>();
    const newTask = vi.fn(() => newFlight.promise);

    void flight.run('old', () => oldFlight.promise);
    const renewed = flight.run('new', newTask);

    oldFlight.resolve('old');
    await Promise.resolve();
    // The new flight must still be the shared in-flight promise.
    expect(flight.run('new', newTask)).toBe(renewed);
    expect(newTask).toHaveBeenCalledTimes(1);
    newFlight.resolve('new');
    await renewed;
  });

  it('clears the flight when the task rejects so the next run retries', async () => {
    const flight = createSingleFlight<string>();
    const failing = vi.fn(() => Promise.reject(new Error('nope')));

    await expect(flight.run('a', failing)).rejects.toThrow('nope');
    await expect(flight.run('a', failing)).rejects.toThrow('nope');
    expect(failing).toHaveBeenCalledTimes(2);
  });
});
