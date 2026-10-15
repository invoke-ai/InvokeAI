import type { GenerationDeviceOption } from '@features/queue/core/deviceLabels';

import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { apiFetchJson, getApiErrorMessage } from '@platform/transport/http';

/**
 * Read available accelerators and server-wide generation_devices without polling; configuration takes effect only
 * after restart and has no socket updates.
 */

/** `auto` uses every available accelerator; an explicit list pins generation to those devices. */
export type GenerationDevicesSetting = 'auto' | string[];

export interface GenerationDevicesSnapshot {
  /** Every installed generation device, in backend order. Empty until loaded. */
  options: GenerationDeviceOption[];
  /** The configured setting, or null while unknown. */
  setting: GenerationDevicesSetting | null;
  loadState: 'idle' | 'loading' | 'loaded' | 'error';
  error: string | null;
}

const EMPTY_SNAPSHOT: GenerationDevicesSnapshot = {
  error: null,
  loadState: 'idle',
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

/**
 * `GET/PATCH /api/v1/app/runtime_config` answer with the config nested under `config`
 * alongside the `set_fields` list — not with the settings at the top level.
 */
interface RuntimeConfigResponse {
  config?: {
    generation_devices?: GenerationDevicesSetting | null;
  } | null;
  set_fields?: string[];
}

const readGenerationDevices = (response: RuntimeConfigResponse | null): GenerationDevicesSetting | null =>
  response?.config?.generation_devices ?? null;

/**
 * Non-admins can read device options but not runtime config; return a null setting without failing the read-only
 * view.
 */
export const refreshGenerationDevices = (): Promise<void> => {
  if (inflight) {
    return inflight;
  }

  const owner = captureAccountScope();

  if (store.getSnapshot().loadState === 'idle') {
    store.patchSnapshot({ loadState: 'loading' });
  }

  const refresh = Promise.all([
    apiFetchJson<GenerationDeviceOption[]>('/api/v1/app/generation_device_options', { signal: owner.signal }),
    apiFetchJson<RuntimeConfigResponse>('/api/v1/app/runtime_config', { signal: owner.signal }).catch(() => null),
  ])
    .then(([options, runtimeConfig]) => {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      store.patchSnapshot({
        error: null,
        loadState: 'loaded',
        options,
        setting: readGenerationDevices(runtimeConfig),
      });
    })
    .catch((error: unknown) => {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      store.patchSnapshot({
        error: getApiErrorMessage(error, 'Failed to load generation devices'),
        loadState: 'error',
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

export const getGenerationDevicesSnapshot = (): GenerationDevicesSnapshot => store.getSnapshot();

export const useGenerationDevices = (): GenerationDevicesSnapshot => store.useSnapshot();

/** Just the device options, for label lookups that do not care about the setting. */
export const useGenerationDeviceOptions = (): GenerationDeviceOption[] =>
  store.useSelector((snapshot) => snapshot.options);
