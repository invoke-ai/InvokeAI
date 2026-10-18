import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  apiFetchJson: vi.fn(),
}));

vi.mock('@platform/transport/http', () => ({
  apiFetchJson: (...args: unknown[]) => mocks.apiFetchJson(...args),
  getApiErrorMessage: (_error: unknown, fallback: string) => fallback,
}));

const { getGenerationDevicesSnapshot, refreshGenerationDevices, updateGenerationDevices, updateMaxQueueHistory } =
  await import('./generationDevicesStore');

const options = [
  { device: 'cuda:0', name: 'RTX 5090' },
  { device: 'cuda:1', name: 'RTX 4090' },
];

describe('generationDevicesStore', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
  });

  it('loads the device options together with the configured setting', async () => {
    mocks.apiFetchJson.mockImplementation((url: string) =>
      url.includes('generation_device_options')
        ? Promise.resolve(options)
        : Promise.resolve({ config: { generation_devices: ['cuda:0'] }, set_fields: ['generation_devices'] })
    );

    await refreshGenerationDevices();

    expect(getGenerationDevicesSnapshot()).toMatchObject({
      loadState: 'loaded',
      options,
      setting: ['cuda:0'],
    });
  });

  it('still reports the options when the admin-only runtime config is forbidden', async () => {
    // The options endpoint is open to any authenticated user; runtime_config is
    // admin-only. A non-admin must get a read-only view, not an error state.
    mocks.apiFetchJson.mockImplementation((url: string) =>
      url.includes('generation_device_options') ? Promise.resolve(options) : Promise.reject(new Error('403'))
    );

    await refreshGenerationDevices();

    expect(getGenerationDevicesSnapshot()).toMatchObject({ error: null, loadState: 'loaded', options, setting: null });
  });

  it('reports an error when the options themselves cannot be read', async () => {
    mocks.apiFetchJson.mockRejectedValue(new Error('boom'));

    await refreshGenerationDevices();

    expect(getGenerationDevicesSnapshot()).toMatchObject({
      error: 'Failed to load generation devices',
      loadState: 'error',
    });
  });

  it('PATCHes the runtime config and adopts the value the server echoes back', async () => {
    // Runtime config is nested under config, not returned at top level.
    mocks.apiFetchJson.mockResolvedValue({
      config: { generation_devices: ['cuda:0', 'cuda:1'] },
      set_fields: ['generation_devices'],
    });

    await updateGenerationDevices(['cuda:0', 'cuda:1']);

    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/app/runtime_config', {
      body: JSON.stringify({ generation_devices: ['cuda:0', 'cuda:1'] }),
      method: 'PATCH',
      signal: expect.anything(),
    });
    expect(getGenerationDevicesSnapshot().setting).toEqual(['cuda:0', 'cuda:1']);
  });

  it('propagates a save failure instead of reporting success', async () => {
    mocks.apiFetchJson.mockRejectedValue(new Error('422 invalid device'));

    await expect(updateGenerationDevices(['cuda:9'])).rejects.toThrow('422 invalid device');
  });

  it('reads the queue history limit, telling "keep everything" apart from "not readable"', async () => {
    mocks.apiFetchJson.mockImplementation((url: string) =>
      url.includes('generation_device_options')
        ? Promise.resolve(options)
        : Promise.resolve({ config: { generation_devices: 'auto', max_queue_history: 0 }, set_fields: [] })
    );
    await refreshGenerationDevices();
    expect(getGenerationDevicesSnapshot().maxQueueHistory).toBe(0);

    mocks.apiFetchJson.mockImplementation((url: string) =>
      url.includes('generation_device_options')
        ? Promise.resolve(options)
        : Promise.resolve({ config: { generation_devices: 'auto', max_queue_history: null }, set_fields: [] })
    );
    await refreshGenerationDevices();
    expect(getGenerationDevicesSnapshot().maxQueueHistory).toBeNull();

    mocks.apiFetchJson.mockImplementation((url: string) =>
      url.includes('generation_device_options') ? Promise.resolve(options) : Promise.reject(new Error('403'))
    );
    await refreshGenerationDevices();
    expect(getGenerationDevicesSnapshot().maxQueueHistory).toBeUndefined();

    mocks.apiFetchJson.mockImplementation((url: string) =>
      url.includes('generation_device_options')
        ? Promise.resolve(options)
        : Promise.resolve({ config: { generation_devices: 'auto', max_queue_history: -3 }, set_fields: [] })
    );
    await refreshGenerationDevices();
    expect(getGenerationDevicesSnapshot().maxQueueHistory).toBeUndefined();
  });

  it('still reads the queue history limit when the device listing fails', async () => {
    mocks.apiFetchJson.mockImplementation((url: string) =>
      url.includes('generation_device_options')
        ? Promise.reject(new Error('500'))
        : Promise.resolve({ config: { generation_devices: 'auto', max_queue_history: 12 }, set_fields: [] })
    );

    await refreshGenerationDevices();

    expect(getGenerationDevicesSnapshot()).toMatchObject({ loadState: 'error', maxQueueHistory: 12 });
  });

  it('PATCHes only the queue history limit so the device selection is left untouched', async () => {
    mocks.apiFetchJson.mockResolvedValue({
      config: { generation_devices: ['cuda:1'], max_queue_history: 25 },
      set_fields: ['generation_devices', 'max_queue_history'],
    });

    await updateMaxQueueHistory(25);

    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/app/runtime_config', {
      body: JSON.stringify({ max_queue_history: 25 }),
      method: 'PATCH',
      signal: expect.anything(),
    });
    expect(getGenerationDevicesSnapshot().maxQueueHistory).toBe(25);

    mocks.apiFetchJson.mockResolvedValue({ config: { generation_devices: ['cuda:1'], max_queue_history: null } });
    await updateMaxQueueHistory(null);
    const lastInit = mocks.apiFetchJson.mock.lastCall?.[1] as RequestInit | undefined;
    expect(JSON.parse(String(lastInit?.body))).toEqual({ max_queue_history: null });
    expect(getGenerationDevicesSnapshot().maxQueueHistory).toBeNull();

    // The server's answer is what is stored, even when it differs from the request.
    mocks.apiFetchJson.mockResolvedValue({ config: { generation_devices: ['cuda:1'], max_queue_history: null } });
    await updateMaxQueueHistory(40);
    expect(getGenerationDevicesSnapshot().maxQueueHistory).toBeNull();

    mocks.apiFetchJson.mockResolvedValue({ detail: 'unexpected' });
    await updateMaxQueueHistory(40);
    expect(getGenerationDevicesSnapshot().maxQueueHistory).toBeUndefined();
  });
});
