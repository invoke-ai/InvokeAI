import { describe, expect, it, vi } from 'vitest';

import { createSingleFlight, createTrailingSingleFlight } from './singleFlight';

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

describe('platform createTrailingSingleFlight', () => {
  it('runs immediately when idle and shares the flight with mid-flight callers', async () => {
    const flight = createTrailingSingleFlight();
    const { promise, resolve } = deferred<void>();
    const task = vi.fn(() => promise);

    const first = flight.run(task);
    const joined = flight.run(task);

    expect(joined).toBe(first);
    expect(flight.inflight()).toBe(first);
    expect(task).toHaveBeenCalledTimes(1);
    resolve();
    await first;
  });

  it('reruns exactly once no matter how many callers joined mid-flight', async () => {
    const flight = createTrailingSingleFlight();
    const initial = deferred<void>();
    const task = vi.fn().mockReturnValueOnce(initial.promise).mockResolvedValue(undefined);

    void flight.run(task);
    void flight.run(task);
    void flight.run(task);
    void flight.run(task);

    initial.resolve();
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });

    expect(task).toHaveBeenCalledTimes(2);
    expect(flight.inflight()).toBeNull();
  });

  it('does not rerun when no caller joined mid-flight', async () => {
    const flight = createTrailingSingleFlight();
    const task = vi.fn(() => Promise.resolve());

    await flight.run(task);
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });

    expect(task).toHaveBeenCalledTimes(1);
  });

  it('reset cancels the in-flight run and any queued rerun', async () => {
    const flight = createTrailingSingleFlight();
    const initial = deferred<void>();
    const task = vi.fn().mockReturnValueOnce(initial.promise).mockResolvedValue(undefined);

    void flight.run(task);
    void flight.run(task);
    flight.reset();

    expect(flight.inflight()).toBeNull();
    initial.resolve();
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });

    expect(task).toHaveBeenCalledTimes(1);
  });

  it('swallows a rerun rejection but propagates the joined flight rejection', async () => {
    const flight = createTrailingSingleFlight();
    const initial = deferred<void>();
    const task = vi.fn().mockReturnValueOnce(initial.promise).mockRejectedValue(new Error('rerun failed'));

    const first = flight.run(task);
    const joined = flight.run(task);

    initial.resolve();
    await expect(joined).resolves.toBeUndefined();
    await expect(first).resolves.toBeUndefined();
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });

    expect(task).toHaveBeenCalledTimes(2);
    expect(flight.inflight()).toBeNull();
  });

  it('turns a synchronously-throwing task into a rejected flight', async () => {
    const flight = createTrailingSingleFlight();
    const task = vi.fn(() => {
      throw new Error('sync throw');
    });

    await expect(flight.run(task as () => Promise<void>)).rejects.toThrow('sync throw');
    expect(flight.inflight()).toBeNull();
  });
});
