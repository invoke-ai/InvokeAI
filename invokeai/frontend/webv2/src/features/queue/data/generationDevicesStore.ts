import type { GenerationDeviceOption } from '@features/queue/core/deviceLabels';

import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { apiFetchJson, getApiErrorMessage } from '@platform/transport/http';

/** No polling: configuration takes effect only after restart and has no socket updates. */

/** `auto` uses every available accelerator; an explicit list pins generation to those devices. */
export type GenerationDevicesSetting = 'auto' | string[];

/** `null` when the server prunes nothing; `undefined` while unknown (not loaded, or not an administrator). */
export type MaxQueueHistorySetting = number | null;

export interface GenerationDevicesSnapshot {
  options: GenerationDeviceOption[];
  setting: GenerationDevicesSetting | null;
  maxQueueHistory: MaxQueueHistorySetting | undefined;
  loadState: 'idle' | 'loading' | 'loaded' | 'error';
  error: string | null;
}

const EMPTY_SNAPSHOT: GenerationDevicesSnapshot = {
  error: null,
  loadState: 'idle',
  maxQueueHistory: undefined,
  options: [],
  setting: null,
};

const store = createExternalStore<GenerationDevicesSnapshot>(EMPTY_SNAPSHOT);

let inflight: Promise<void> | null = null;

registerAccountOwnedResource({
  clear: () => {
    inflight = null;
    store.setSnapshot(EMPTY_SNAPSHOT);
  },
  name: 'generation-devices',
});

/** The config nests under `config` beside `set_fields`, not at the top level. */
interface RuntimeConfigResponse {
  config?: {
    generation_devices?: GenerationDevicesSetting | null;
    max_queue_history?: number | null;
  } | null;
  set_fields?: string[];
}

const readGenerationDevices = (response: RuntimeConfigResponse | null): GenerationDevicesSetting | null =>
  response?.config?.generation_devices ?? null;

/** An unexpected value reads as unreadable: shown as keep-all it would invite saving over the real limit. */
const readMaxQueueHistory = (response: RuntimeConfigResponse | null): MaxQueueHistorySetting | undefined => {
  const value = response?.config?.max_queue_history;

  return value === null || (typeof value === 'number' && Number.isSafeInteger(value) && value >= 0) ? value : undefined;
};

/** Non-admins can read device options but not runtime config. */
export const refreshGenerationDevices = (): Promise<void> => {
  if (inflight) {
    return inflight;
  }

  const owner = captureAccountScope();

  if (store.getSnapshot().loadState === 'idle') {
    store.patchSnapshot({ loadState: 'loading' });
  }

  // The runtime config is read on its own: a failed device listing must not hide the queue history setting.
  const runtimeConfigRead = apiFetchJson<RuntimeConfigResponse>('/api/v1/app/runtime_config', {
    signal: owner.signal,
  }).catch(() => null);
  const refresh = Promise.all([
    apiFetchJson<GenerationDeviceOption[]>('/api/v1/app/generation_device_options', { signal: owner.signal }),
    runtimeConfigRead,
  ])
    .then(([options, runtimeConfig]) => {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      store.patchSnapshot({
        error: null,
        loadState: 'loaded',
        maxQueueHistory: readMaxQueueHistory(runtimeConfig),
        options,
        setting: readGenerationDevices(runtimeConfig),
      });
    })
    .catch(async (error: unknown) => {
      const runtimeConfig = await runtimeConfigRead;
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      store.patchSnapshot({
        error: getApiErrorMessage(error, 'Failed to load generation devices'),
        loadState: 'error',
        maxQueueHistory: readMaxQueueHistory(runtimeConfig),
      });
    })
    .finally(() => {
      if (inflight === refresh) {
        inflight = null;
      }
    });

  inflight = refresh;

  return inflight;
};

/** Admin changes take effect on restart; backend device validation rejects stale or nonexistent selections. */
export const updateGenerationDevices = async (setting: GenerationDevicesSetting): Promise<void> => {
  const owner = captureAccountScope();

  const runtimeConfig = await apiFetchJson<RuntimeConfigResponse>('/api/v1/app/runtime_config', {
    body: JSON.stringify({ generation_devices: setting }),
    method: 'PATCH',
    signal: owner.signal,
  });

  assertAccountScopeCurrent(owner);

  store.patchSnapshot({ error: null, setting: readGenerationDevices(runtimeConfig) ?? setting });
};

/** Sends only `max_queue_history`; stores what the server answers, and an unreadable answer leaves it unknown. */
export const updateMaxQueueHistory = async (value: MaxQueueHistorySetting): Promise<void> => {
  const owner = captureAccountScope();

  const runtimeConfig = await apiFetchJson<RuntimeConfigResponse>('/api/v1/app/runtime_config', {
    body: JSON.stringify({ max_queue_history: value }),
    method: 'PATCH',
    signal: owner.signal,
  });

  assertAccountScopeCurrent(owner);

  store.patchSnapshot({ maxQueueHistory: readMaxQueueHistory(runtimeConfig) });
};

export const getGenerationDevicesSnapshot = (): GenerationDevicesSnapshot => store.getSnapshot();

export const useGenerationDevices = (): GenerationDevicesSnapshot => store.useSnapshot();

export const useGenerationDeviceOptions = (): GenerationDeviceOption[] =>
  store.useSelector((snapshot) => snapshot.options);
