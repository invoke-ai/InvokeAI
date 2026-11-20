import { accountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  apiFetchJson: vi.fn(),
}));

vi.mock('@platform/transport/http', () => ({
  apiFetchJson: mocks.apiFetchJson,
}));

import {
  DEFAULT_REMOTE_WORKERS_SETTINGS,
  getRemoteWorkerName,
  getRemoteWorkerUrls,
  getRemoteWorkersSettings,
  loadRemoteWorkersSettings,
  setRemoteWorkerName,
  setRemoteWorkersSettings,
} from './remoteWorkersStore';

let resetCounter = 0;

const activateSingleUser = (): void => {
  accountLifecycle.activate(`remote-worker-store-reset-${++resetCounter}`);
  accountLifecycle.activate('single-user');
};

describe('remote worker server settings', () => {
  beforeEach(async () => {
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchJson.mockImplementation((_path: string, init?: RequestInit) => {
      if (init?.method === 'PUT') {
        return Promise.resolve(JSON.parse(String(init.body)));
      }
      return Promise.resolve({ ...DEFAULT_REMOTE_WORKERS_SETTINGS, disabledWorkerUrls: [], workerNames: {} });
    });
    activateSingleUser();
    await loadRemoteWorkersSettings();
  });

  afterEach(() => {
    accountLifecycle.invalidate();
  });

  it('loads settings from the authenticated primary server', async () => {
    mocks.apiFetchJson.mockImplementation((_path: string, init?: RequestInit) => {
      if (init?.method === 'PUT') {
        return Promise.resolve(JSON.parse(String(init.body)));
      }
      return Promise.resolve({
        ...DEFAULT_REMOTE_WORKERS_SETTINGS,
        enabled: true,
        dispatchMode: 'remote_only',
        workerUrls: 'https://example.test/invoke',
      });
    });

    await loadRemoteWorkersSettings();

    expect(getRemoteWorkersSettings().dispatchMode).toBe('remote_only');
    expect(getRemoteWorkersSettings().workerUrls).toBe('https://example.test/invoke');
  });

  it('does not depend on browser localStorage and strips embedded URL credentials', () => {
    Object.defineProperty(globalThis, 'localStorage', {
      configurable: true,
      get: () => {
        throw new Error('remote worker settings must not read browser storage');
      },
    });

    setRemoteWorkersSettings({
      workerUrls: 'http://alice:secret@192.168.1.101:9090\nhttp://192.168.1.102:9090',
    });

    expect(getRemoteWorkersSettings().workerUrls).toBe('http://192.168.1.101:9090\nhttp://192.168.1.102:9090');
    expect(getRemoteWorkerUrls(getRemoteWorkersSettings().workerUrls)).toEqual([
      'http://192.168.1.101:9090',
      'http://192.168.1.102:9090',
    ]);

    Reflect.deleteProperty(globalThis, 'localStorage');
  });

  it('stores a user-defined worker name by normalized URL', () => {
    const url = 'http://192.168.1.101:9090';

    setRemoteWorkerName(url, 'RTX5080');

    expect(getRemoteWorkerName(url, 0)).toBe('RTX5080');
    expect(getRemoteWorkersSettings().workerNames[url]).toBe('RTX5080');

    setRemoteWorkerName(url, '   ');
    expect(getRemoteWorkerName(url, 0)).toBe('Remote 1');
  });
});
