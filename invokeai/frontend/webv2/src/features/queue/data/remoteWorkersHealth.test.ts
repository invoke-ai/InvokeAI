import { accountLifecycle } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({ apiFetchJson: vi.fn() }));
vi.mock('@platform/transport/http', () => transport);

import {
  invalidateRemoteWorkerHealth,
  refreshRemoteWorkerHealth,
  remoteWorkersHealthStore,
} from './remoteWorkersHealth';
import {
  DEFAULT_REMOTE_WORKERS_SETTINGS,
  getRemoteWorkerUrls,
  getRemoteWorkersSettings,
  isRemoteWorkerEnabled,
  setRemoteWorkerEnabled,
  setRemoteWorkersSettings,
} from './remoteWorkersStore';

const worker1 = 'http://192.168.1.101:9090';
const worker2 = 'http://192.168.1.102:9090';

describe('Remote worker health', () => {
  beforeEach(() => {
    accountLifecycle.activate('worker-health-test-user');
    setRemoteWorkersSettings({
      disabledWorkerUrls: [],
      enabled: true,
      workerUrls: `${worker1}\n${worker2}`,
    });
    remoteWorkersHealthStore.setSnapshot({ byUrl: {} });
    transport.apiFetchJson.mockReset();
  });

  it('does not ping any worker while distributed rendering is disabled', async () => {
    setRemoteWorkersSettings({ enabled: false });

    await refreshRemoteWorkerHealth([worker1, worker2]);

    expect(transport.apiFetchJson).not.toHaveBeenCalled();
    expect(remoteWorkersHealthStore.getSnapshot().byUrl).toEqual({});
  });

  it('does not ping a worker disabled with its power toggle', async () => {
    setRemoteWorkerEnabled(worker1, false);
    transport.apiFetchJson.mockResolvedValue({ status: 'online' });

    await refreshRemoteWorkerHealth([worker1, worker2]);

    expect(transport.apiFetchJson).toHaveBeenCalledTimes(1);
    expect(String(transport.apiFetchJson.mock.calls[0]?.[0])).toContain(encodeURIComponent(worker2));
    expect(remoteWorkersHealthStore.getSnapshot().byUrl[worker1]).toBeUndefined();
    expect(remoteWorkersHealthStore.getSnapshot().byUrl[worker2]?.status).toBe('online');
  });

  it('ignores an in-flight result if the worker is disabled before it completes', async () => {
    let finish!: (value: { status: string }) => void;
    transport.apiFetchJson.mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        })
    );

    const pending = refreshRemoteWorkerHealth([worker1]);
    await vi.waitFor(() => expect(transport.apiFetchJson).toHaveBeenCalledTimes(1));
    setRemoteWorkerEnabled(worker1, false);
    finish({ status: 'online' });
    await pending;

    expect(remoteWorkersHealthStore.getSnapshot().byUrl[worker1]?.status).not.toBe('online');
  });

  it('reports an offline worker online after recovery', async () => {
    transport.apiFetchJson.mockResolvedValueOnce({ status: 'offline' }).mockResolvedValueOnce({ status: 'online' });

    await refreshRemoteWorkerHealth([worker1]);
    expect(remoteWorkersHealthStore.getSnapshot().byUrl[worker1]?.status).toBe('offline');

    remoteWorkersHealthStore.setSnapshot({ byUrl: { [worker1]: { status: 'offline', checkedAt: 0 } } });
    await refreshRemoteWorkerHealth([worker1]);

    expect(remoteWorkersHealthStore.getSnapshot().byUrl[worker1]?.status).toBe('online');
  });

  it('stores the disabled choice per URL and retains it across the global toggle', () => {
    setRemoteWorkerEnabled(worker1, false);

    expect(getRemoteWorkersSettings().disabledWorkerUrls).toEqual([worker1]);
    expect(getRemoteWorkerUrls(`${worker2}\n${worker1}`).filter(isRemoteWorkerEnabled)).toEqual([worker2]);

    setRemoteWorkersSettings({ enabled: false });
    setRemoteWorkersSettings({ enabled: true });

    expect(isRemoteWorkerEnabled(worker1)).toBe(false);
    setRemoteWorkerEnabled(worker1, true);
    expect(getRemoteWorkersSettings().disabledWorkerUrls).toEqual([]);
  });

  it('forces a fresh worker probe after credential health is invalidated', async () => {
    remoteWorkersHealthStore.setSnapshot({
      byUrl: { [worker1]: { status: 'login_required', checkedAt: Date.now() } },
    });
    transport.apiFetchJson.mockResolvedValue({ status: 'online' });

    await refreshRemoteWorkerHealth([worker1]);
    expect(transport.apiFetchJson).not.toHaveBeenCalled();

    invalidateRemoteWorkerHealth(worker1);
    await refreshRemoteWorkerHealth([worker1]);

    expect(transport.apiFetchJson).toHaveBeenCalledTimes(1);
    expect(remoteWorkersHealthStore.getSnapshot().byUrl[worker1]?.status).toBe('online');
  });

  it('does not mark a worker online when its saved login is rejected', async () => {
    transport.apiFetchJson.mockResolvedValue({ status: 'login_required' });

    await refreshRemoteWorkerHealth([worker1]);

    expect(remoteWorkersHealthStore.getSnapshot().byUrl[worker1]?.status).toBe('login_required');
  });

  it('discards results from a previous account', async () => {
    let finish!: (value: { status: string }) => void;
    transport.apiFetchJson.mockImplementation((path: string) => {
      if (path === '/api/v1/remote_workers/settings') {
        return Promise.resolve({
          ...DEFAULT_REMOTE_WORKERS_SETTINGS,
          disabledWorkerUrls: [],
          workerNames: {},
        });
      }
      return new Promise((resolve) => {
        finish = resolve;
      });
    });

    const pending = refreshRemoteWorkerHealth([worker1]);
    accountLifecycle.activate('another-user');
    finish({ status: 'online' });
    await pending;

    expect(remoteWorkersHealthStore.getSnapshot().byUrl[worker1]).toBeUndefined();
  });
});
