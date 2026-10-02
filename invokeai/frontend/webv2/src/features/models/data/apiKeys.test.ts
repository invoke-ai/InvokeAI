import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const identity = vi.hoisted(() => ({ getUserStorageScope: vi.fn(() => ':user:a') }));

vi.mock('@features/identity', () => identity);

import {
  clearCivitaiApiKey,
  getAccessTokenForSource,
  getCivitaiApiKey,
  isCivitaiUrl,
  setCivitaiApiKey,
} from './apiKeys';

const storage = new Map<string, string>();

beforeEach(() => {
  identity.getUserStorageScope.mockReturnValue(':user:a');
  vi.stubGlobal('window', {
    localStorage: {
      getItem: (key: string) => storage.get(key) ?? null,
      removeItem: (key: string) => storage.delete(key),
      setItem: (key: string, value: string) => storage.set(key, value),
    },
  });
});

afterEach(() => {
  storage.clear();
  vi.unstubAllGlobals();
});

describe('isCivitaiUrl', () => {
  it('matches the civitai host and its subdomains only', () => {
    expect(isCivitaiUrl('https://civitai.com/models/1')).toBe(true);
    expect(isCivitaiUrl('https://api.civitai.com/download')).toBe(true);
    expect(isCivitaiUrl('https://not-civitai.com/models/1')).toBe(false);
    expect(isCivitaiUrl('https://civitai.com.evil.example/models/1')).toBe(false);
  });

  it('rejects malformed URLs and plain paths', () => {
    expect(isCivitaiUrl('civitai.com/models/1')).toBe(false);
    expect(isCivitaiUrl('/models/file.safetensors')).toBe(false);
    expect(isCivitaiUrl('')).toBe(false);
  });
});

describe('civitai key storage', () => {
  it('stores, reads, and clears the key per user scope', () => {
    expect(getCivitaiApiKey()).toBeNull();

    setCivitaiApiKey('key-a');
    expect(getCivitaiApiKey()).toBe('key-a');

    identity.getUserStorageScope.mockReturnValue(':user:b');
    expect(getCivitaiApiKey()).toBeNull();

    identity.getUserStorageScope.mockReturnValue(':user:a');
    clearCivitaiApiKey();
    expect(getCivitaiApiKey()).toBeNull();
  });

  it('returns null instead of throwing when storage is unavailable', () => {
    vi.stubGlobal('window', {
      localStorage: {
        getItem: () => {
          throw new Error('denied');
        },
        setItem: () => {
          throw new Error('denied');
        },
      },
    });

    expect(getCivitaiApiKey()).toBeNull();
    expect(() => setCivitaiApiKey('key')).not.toThrow();
  });
});

describe('getAccessTokenForSource', () => {
  it('returns the saved key only for civitai URLs', () => {
    setCivitaiApiKey('key-a');
    expect(getAccessTokenForSource('https://civitai.com/api/download/models/1')).toBe('key-a');
    expect(getAccessTokenForSource('https://huggingface.co/owner/repo')).toBeUndefined();
  });

  it('returns undefined when no key is saved', () => {
    expect(getAccessTokenForSource('https://civitai.com/api/download/models/1')).toBeUndefined();
  });
});
