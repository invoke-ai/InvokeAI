import { createElement } from 'react';
import { renderToString } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';

import { createCollectionStore, createExternalStore, createKeyedTransientStore } from './externalStore';

describe('platform createExternalStore', () => {
  it('does not notify listeners for identical snapshots or empty patches', () => {
    const initial = { count: 0, label: 'idle' };
    const store = createExternalStore(initial);
    const listener = vi.fn();

    store.subscribe(listener);

    store.setSnapshot(initial);
    store.setSnapshot({ count: 0, label: 'idle' });
    store.patchSnapshot({});
    store.patchSnapshot({ count: 0 });

    expect(listener).not.toHaveBeenCalled();
  });

  it('notifies when a patch adds an undefined optional key', () => {
    const store = createExternalStore<{ count: number; label?: string }>({ count: 0 });
    const listener = vi.fn();

    store.subscribe(listener);
    store.patchSnapshot({ label: undefined });

    expect(listener).toHaveBeenCalledTimes(1);
    expect(Object.prototype.hasOwnProperty.call(store.getSnapshot(), 'label')).toBe(true);
  });
});

describe('createKeyedTransientStore', () => {
  it('provides a server snapshot for server rendering', () => {
    const store = createKeyedTransientStore<string, number>();
    store.set('answer', 42);

    const View = () => createElement('span', null, store.useValue('answer'));

    expect(renderToString(createElement(View))).toBe('<span>42</span>');
  });

  it('notifies only listeners for the changed key and global listeners', () => {
    const store = createKeyedTransientStore<string, number>();
    const keyListener = vi.fn();
    const otherKeyListener = vi.fn();
    const globalListener = vi.fn();

    store.subscribeKey('a', keyListener);
    store.subscribeKey('b', otherKeyListener);
    store.subscribe(globalListener);

    store.set('a', 1);

    expect(keyListener).toHaveBeenCalledTimes(1);
    expect(otherKeyListener).not.toHaveBeenCalled();
    expect(globalListener).toHaveBeenCalledTimes(1);
    expect(store.get('a')).toBe(1);
  });

  it('does not notify when setting an identical value', () => {
    const store = createKeyedTransientStore<string, number>();
    const listener = vi.fn();

    store.set('a', 1);
    store.subscribeKey('a', listener);
    store.set('a', 1);

    expect(listener).not.toHaveBeenCalled();
  });
});

describe('createCollectionStore', () => {
  it('returns a stable list snapshot until registrations change', () => {
    const store = createCollectionStore<{ id: string }>();

    const empty = store.list();

    expect(store.list()).toBe(empty);

    const dispose = store.register({ id: 'first' }, 'first');
    const populated = store.list();

    expect(populated).not.toBe(empty);
    expect(store.list()).toBe(populated);

    dispose();

    expect(store.list()).not.toBe(populated);
    expect(store.list()).toEqual([]);
  });

  it('replaces duplicate logical registrations and only the active disposer can clear them', () => {
    const store = createCollectionStore<{ id: string; value: string }>();
    const disposeFirst = store.register({ id: 'duplicate', value: 'first' }, 'duplicate');
    const disposeSecond = store.register({ id: 'duplicate', value: 'second' }, 'duplicate');

    expect(store.list()).toEqual([{ id: 'duplicate', value: 'second' }]);
    expect(store.findLatest((item) => item.id === 'duplicate')?.value).toBe('second');

    disposeFirst();

    expect(store.findLatest((item) => item.id === 'duplicate')?.value).toBe('second');

    disposeSecond();

    expect(store.findLatest((item) => item.id === 'duplicate')).toBeUndefined();
  });

  it('keeps registrations with different logical keys side by side', () => {
    const store = createCollectionStore<{ id: string; value: string }>();
    const disposeFirst = store.register({ id: 'duplicate', value: 'first' }, 'duplicate:first');
    const disposeSecond = store.register({ id: 'duplicate', value: 'second' }, 'duplicate:second');

    expect(store.list()).toEqual([
      { id: 'duplicate', value: 'first' },
      { id: 'duplicate', value: 'second' },
    ]);

    disposeFirst();

    expect(store.list()).toEqual([{ id: 'duplicate', value: 'second' }]);

    disposeSecond();

    expect(store.findLatest((item) => item.id === 'duplicate')).toBeUndefined();
  });
});
