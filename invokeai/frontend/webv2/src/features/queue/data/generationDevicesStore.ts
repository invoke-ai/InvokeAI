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
 * The GPUs available for generation, and which of them the server is configured to
 * use. Pull-based like `modelCacheStore`: there are no socket events for app config.
 *
 * `generation_devices` is server-wide app config (admin-only, `PATCH
 * /api/v1/app/runtime_config`) and only takes effect after a restart, so this is
 * deliberately not treated as live state — nothing here re-reads on a timer.
 */

/** `auto` uses every available GPU; an explicit list pins generation to those devices. */
export type GenerationDevicesSetting = 'auto' | string[];

export interface GenerationDevicesSnapshot {
  /** Every installed device, in CUDA-index order. Empty until loaded. */
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

interface RuntimeConfigResponse {
  generation_devices?: GenerationDevicesSetting | null;
}

/**
 * Load the device options and the current setting together.
 *
 * The options endpoint is available to any authenticated user, but the runtime
 * config is admin-only. A non-admin therefore gets the options and a null setting
 * rather than an error, which is what the read-only view wants.
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
        setting: runtimeConfig?.generation_devices ?? null,
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

/**
 * Persist a new `generation_devices` setting. Admin-only; takes effect on restart.
 *
 * The backend validates that every device actually exists on the machine and 422s
 * otherwise, so a stale options list cannot write a config that fails at startup.
 */
export const updateGenerationDevices = async (setting: GenerationDevicesSetting): Promise<void> => {
  const owner = captureAccountScope();

  const runtimeConfig = await apiFetchJson<RuntimeConfigResponse>('/api/v1/app/runtime_config', {
    body: JSON.stringify({ generation_devices: setting }),
    method: 'PATCH',
    signal: owner.signal,
  });

  assertAccountScopeCurrent(owner);

  store.patchSnapshot({ error: null, setting: runtimeConfig?.generation_devices ?? setting });
};

export const getGenerationDevicesSnapshot = (): GenerationDevicesSnapshot => store.getSnapshot();

export const useGenerationDevices = (): GenerationDevicesSnapshot => store.useSnapshot();

/** Just the device options, for label lookups that do not care about the setting. */
export const useGenerationDeviceOptions = (): GenerationDeviceOption[] =>
  store.useSelector((snapshot) => snapshot.options);
