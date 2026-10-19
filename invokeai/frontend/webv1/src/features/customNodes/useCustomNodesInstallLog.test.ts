import { beforeEach, describe, expect, it } from 'vitest';

import { _resetIdCounter, $installLog, addInstallLogEntry, clearInstallLog } from './useCustomNodesInstallLog';

describe('Install Log Store', () => {
  beforeEach(() => {
    clearInstallLog();
    _resetIdCounter();
  });

  it('starts with an empty log', () => {
    expect($installLog.get()).toEqual([]);
  });

  it('adds an entry to the log', () => {
    addInstallLogEntry({ name: 'test-pack', status: 'installing' });
    const log = $installLog.get();
    expect(log).toHaveLength(1);
    expect(log[0]!.name).toBe('test-pack');
    expect(log[0]!.status).toBe('installing');
    expect(log[0]!.id).toBe('0');
  });

  it('assigns incrementing IDs', () => {
    addInstallLogEntry({ name: 'pack-1', status: 'installing' });
    addInstallLogEntry({ name: 'pack-2', status: 'completed' });
    const log = $installLog.get();
    // Newest first
    expect(log[0]!.id).toBe('1');
    expect(log[1]!.id).toBe('0');
  });

  it('prepends new entries (newest first)', () => {
    addInstallLogEntry({ name: 'first', status: 'installing' });
    addInstallLogEntry({ name: 'second', status: 'completed' });
    addInstallLogEntry({ name: 'third', status: 'error' });
    const log = $installLog.get();
    expect(log[0]!.name).toBe('third');
    expect(log[1]!.name).toBe('second');
    expect(log[2]!.name).toBe('first');
  });

  it('includes a timestamp', () => {
    const before = Date.now();
    addInstallLogEntry({ name: 'pack', status: 'installing' });
    const after = Date.now();
    const entry = $installLog.get()[0]!;
    expect(entry.timestamp).toBeGreaterThanOrEqual(before);
    expect(entry.timestamp).toBeLessThanOrEqual(after);
  });

  it('preserves the message field', () => {
    addInstallLogEntry({ name: 'pack', status: 'error', message: 'Something went wrong' });
    expect($installLog.get()[0]!.message).toBe('Something went wrong');
  });

  it('allows message to be undefined', () => {
    addInstallLogEntry({ name: 'pack', status: 'completed' });
    expect($installLog.get()[0]!.message).toBeUndefined();
  });

  it('clears the log', () => {
    addInstallLogEntry({ name: 'pack-1', status: 'installing' });
    addInstallLogEntry({ name: 'pack-2', status: 'completed' });
    expect($installLog.get()).toHaveLength(2);

    clearInstallLog();
    expect($installLog.get()).toEqual([]);
  });

  it('supports all status types', () => {
    const statuses = ['installing', 'completed', 'error', 'uninstalled'] as const;
    for (const status of statuses) {
      addInstallLogEntry({ name: `pack-${status}`, status });
    }
    const log = $installLog.get();
    expect(log).toHaveLength(4);
    expect(log.map((e) => e.status).sort()).toEqual(['completed', 'error', 'installing', 'uninstalled']);
  });

  it('returns the created entry', () => {
    const entry = addInstallLogEntry({ name: 'my-pack', status: 'installing' });
    expect(entry.name).toBe('my-pack');
    expect(entry.id).toBeDefined();
    expect(entry.timestamp).toBeDefined();
  });
});
