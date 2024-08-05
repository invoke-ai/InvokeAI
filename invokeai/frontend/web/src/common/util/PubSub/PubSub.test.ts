import { PubSub } from 'common/util/PubSub/PubSub';
import { describe, expect, it, vi } from 'vitest';

describe('PubSub', () => {
  it('should call listener when value is published and value changes', () => {
    const pubsub = new PubSub<number>(1);
    const listener = vi.fn();

    pubsub.subscribe(listener);
    pubsub.publish(42);

    expect(listener).toHaveBeenCalledWith(42, 1);
  });

  it('should not call listener if value does not change', () => {
    const pubsub = new PubSub<number>(42);
    const listener = vi.fn();

    pubsub.subscribe(listener);
    pubsub.publish(42);

    expect(listener).not.toHaveBeenCalled();
  });

  it('should handle non-primitive values', () => {
    const pubsub = new PubSub<{ foo: string }>({ foo: 'bar' });
    const listener = vi.fn();

    pubsub.subscribe(listener);
    pubsub.publish({ foo: 'bar' });

    expect(listener).toHaveBeenCalled();
  });

  it('should call listener with old and new value', () => {
    const pubsub = new PubSub<number>(1);
    const listener = vi.fn();

    pubsub.subscribe(listener);
    pubsub.publish(2);

    expect(listener).toHaveBeenCalledWith(2, 1);
  });

  it('should allow unsubscribing', () => {
    const pubsub = new PubSub<number>(1);
    const listener1 = vi.fn();
    const listener2 = vi.fn();

    const unsubscribe = pubsub.subscribe(listener1);
    pubsub.subscribe(listener2);
    unsubscribe();
    pubsub.publish(42);

    expect(listener1).not.toHaveBeenCalled();
    expect(listener2).toHaveBeenCalled();
    expect(pubsub.getListeners().size).toBe(1);
  });

  it('should clear all listeners', () => {
    const pubsub = new PubSub<number>(1);
    const listener1 = vi.fn();
    const listener2 = vi.fn();

    pubsub.subscribe(listener1);
    pubsub.subscribe(listener2);
    pubsub.off();
    pubsub.publish(42);

    expect(listener1).not.toHaveBeenCalled();
    expect(listener2).not.toHaveBeenCalled();
    expect(pubsub.getListeners().size).toBe(0);
  });

  it('should use custom compareFn', () => {
    const compareFn = (a: number, b: number) => Math.abs(a) === Math.abs(b);
    const pubsub = new PubSub<number>(1, compareFn);
    const listener = vi.fn();

    pubsub.subscribe(listener);
    pubsub.publish(-1);

    expect(listener).not.toHaveBeenCalled();
  });

  it('should handle multiple listeners', () => {
    const pubsub = new PubSub<number>(1);
    const listener1 = vi.fn();
    const listener2 = vi.fn();

    pubsub.subscribe(listener1);
    pubsub.subscribe(listener2);
    pubsub.publish(42);

    expect(listener1).toHaveBeenCalledWith(42, 1);
    expect(listener2).toHaveBeenCalledWith(42, 1);
    expect(pubsub.getListeners().size).toBe(2);
  });

  it('should get the current value', () => {
    const pubsub = new PubSub<number>(42);
    expect(pubsub.getValue()).toBe(42);
    pubsub.publish(43);
    expect(pubsub.getValue()).toBe(43);
  });

  it('should get the listeners', () => {
    const pubsub = new PubSub<number>(1);
    const listener1 = vi.fn();
    const listener2 = vi.fn();

    pubsub.subscribe(listener1);
    pubsub.subscribe(listener2);

    expect(pubsub.getListeners()).toEqual(new Set([listener1, listener2]));
  });
});
