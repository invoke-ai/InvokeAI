import { beforeEach, describe, expect, it, vi } from 'vitest';

import { getRecentEntryIds, recordRecentEntry } from './recents';

const STORAGE_KEY = 'invokeai:v7:webv2:palette-recents';
const storage = new Map<string, string>();

const localStorageStub = {
  getItem: (key: string): string | null => storage.get(key) ?? null,
  setItem: (key: string, value: string): void => {
    storage.set(key, value);
  },
};

vi.stubGlobal('window', { localStorage: localStorageStub });

beforeEach(() => {
  storage.clear();
  vi.restoreAllMocks();
});

describe('palette recents', () => {
  it('persists only explicitly durable entries', () => {
    recordRecentEntry({ id: 'provider-result', isPersistentRecent: false });
    expect(storage.has(STORAGE_KEY)).toBe(false);

    recordRecentEntry({ id: 'app.invoke', isPersistentRecent: true });
    expect(getRecentEntryIds()).toEqual(['app.invoke']);
  });

  it('continues to read the legacy plain-id ring buffer', () => {
    storage.set(STORAGE_KEY, JSON.stringify(['legacy.command', 'legacy.setting']));

    expect(getRecentEntryIds()).toEqual(['legacy.command', 'legacy.setting']);
  });

  it('returns no recents when storage access throws', () => {
    vi.spyOn(localStorageStub, 'getItem').mockImplementation(() => {
      throw new DOMException('Storage disabled', 'SecurityError');
    });

    expect(getRecentEntryIds()).toEqual([]);
  });

  it('keeps recording non-fatal when the initial read throws', () => {
    const getItem = vi.spyOn(localStorageStub, 'getItem').mockImplementation(() => {
      throw new DOMException('Storage disabled', 'SecurityError');
    });

    expect(() => recordRecentEntry({ id: 'app.invoke', isPersistentRecent: true })).not.toThrow();

    getItem.mockRestore();
    expect(getRecentEntryIds()).toEqual(['app.invoke']);
  });
});
