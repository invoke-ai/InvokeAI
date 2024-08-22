import { describe, expect, it, vi } from 'vitest';

import { SyncableMap } from './SyncableMap';

describe('SyncableMap', () => {
  it('should initialize with entries', () => {
    const initialEntries = [
      ['key1', 'value1'],
      ['key2', 'value2'],
    ] as const;
    const map = new SyncableMap(initialEntries);
    expect(map.size).toBe(2);
    expect(map.get('key1')).toBe('value1');
    expect(map.get('key2')).toBe('value2');
  });

  it('should notify subscribers when a key is set', () => {
    const map = new SyncableMap<string, string>();
    const subscriber = vi.fn();
    map.subscribe(subscriber);

    map.set('key1', 'value1');

    expect(subscriber).toHaveBeenCalledTimes(1);
    expect(map.get('key1')).toBe('value1');
  });

  it('should notify subscribers when a key is deleted', () => {
    const map = new SyncableMap<string, string>([['key1', 'value1']]);
    const subscriber = vi.fn();
    map.subscribe(subscriber);

    map.delete('key1');

    expect(subscriber).toHaveBeenCalledTimes(1);
    expect(map.get('key1')).toBeUndefined();
  });

  it('should notify subscribers when the map is cleared', () => {
    const map = new SyncableMap<string, string>([
      ['key1', 'value1'],
      ['key2', 'value2'],
    ]);
    const subscriber = vi.fn();
    map.subscribe(subscriber);

    map.clear();

    expect(subscriber).toHaveBeenCalledTimes(1);
    expect(map.size).toBe(0);
  });

  it('should not notify unsubscribed callbacks', () => {
    const map = new SyncableMap<string, string>();
    const subscriber = vi.fn();
    const unsubscribe = map.subscribe(subscriber);

    unsubscribe();

    map.set('key1', 'value1');

    expect(subscriber).not.toHaveBeenCalled();
  });

  it('should return a snapshot of the current state', () => {
    const map = new SyncableMap<string, string>([['key1', 'value1']]);

    const snapshot = map.getSnapshot();

    expect(snapshot.size).toBe(1);
    expect(snapshot.get('key1')).toBe('value1');
  });

  it('should return the same snapshot if there were no changes', () => {
    const map = new SyncableMap<string, string>([['key1', 'value1']]);

    const firstSnapshot = map.getSnapshot();
    const secondSnapshot = map.getSnapshot();

    expect(firstSnapshot).toBe(secondSnapshot);
  });

  it('should return a new snapshot if changes were made', () => {
    const map = new SyncableMap<string, string>([['key1', 'value1']]);

    const firstSnapshot = map.getSnapshot();
    map.set('key2', 'value2');
    const secondSnapshot = map.getSnapshot();

    expect(firstSnapshot).not.toBe(secondSnapshot);
    expect(secondSnapshot.size).toBe(2);
  });

  it('should consider different snapshots unequal', () => {
    const map = new SyncableMap<string, string>([['key1', 'value1']]);

    const firstSnapshot = map.getSnapshot();
    map.set('key2', 'value2');
    const secondSnapshot = map.getSnapshot();

    expect(map['areSnapshotsEqual'](firstSnapshot, secondSnapshot)).toBe(false);
  });

  it('should consider identical snapshots equal', () => {
    const map = new SyncableMap<string, string>([
      ['key1', 'value1'],
      ['key2', 'value2'],
    ]);

    const firstSnapshot = map.getSnapshot();
    const secondSnapshot = map.getSnapshot();

    expect(map['areSnapshotsEqual'](firstSnapshot, secondSnapshot)).toBe(true);
  });
});
