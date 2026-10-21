import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
  type AccountScope,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { apiFetchJson } from '@platform/transport/http';

export type RemoteDispatchMode = 'distributed' | 'remote_only';

export interface RemoteWorkersSettings {
  enabled: boolean;
  dispatchMode: RemoteDispatchMode;
  workerUrls: string;
  /** Optional user-defined display names keyed by normalized worker URL. */
  workerNames: Record<string, string>;
  /** Disabled by normalized URL so reordering workers never changes which worker is paused. */
  disabledWorkerUrls: string[];
  autoTransferMissingModels: boolean;
  keepRemoteCopies: boolean;
  modelTransferHost: string;
}

export const DEFAULT_REMOTE_WORKERS_SETTINGS: RemoteWorkersSettings = {
  enabled: false,
  dispatchMode: 'distributed',
  workerUrls: '',
  workerNames: {},
  disabledWorkerUrls: [],
  autoTransferMissingModels: true,
  keepRemoteCopies: false,
  modelTransferHost: '',
};

const makeDefaultSettings = (): RemoteWorkersSettings => ({
  ...DEFAULT_REMOTE_WORKERS_SETTINGS,
  workerNames: {},
  disabledWorkerUrls: [],
});

const stripEmbeddedUrlCredentials = (raw: string): string =>
  raw.replace(/[^;,\r\n\s]+/g, (candidate) => {
    let url: URL;
    try {
      url = new URL(candidate);
    } catch {
      return candidate;
    }
    if (!['http:', 'https:'].includes(url.protocol) || (!url.username && !url.password)) {
      return candidate;
    }
    const hadTrailingSlash = candidate.endsWith('/');
    url.username = '';
    url.password = '';
    const sanitized = url.toString();
    return !hadTrailingSlash && url.pathname === '/' && !url.search && !url.hash
      ? sanitized.replace(/\/$/, '')
      : sanitized;
  });

const normalizeSettings = (value: unknown): RemoteWorkersSettings => {
  if (typeof value !== 'object' || value === null) {
    return makeDefaultSettings();
  }
  const entry = value as Record<string, unknown>;
  return {
    enabled: entry.enabled === true,
    dispatchMode: entry.dispatchMode === 'remote_only' ? 'remote_only' : 'distributed',
    workerUrls: stripEmbeddedUrlCredentials(typeof entry.workerUrls === 'string' ? entry.workerUrls : ''),
    workerNames:
      typeof entry.workerNames === 'object' && entry.workerNames !== null
        ? Object.fromEntries(
            Object.entries(entry.workerNames as Record<string, unknown>)
              .filter((item): item is [string, string] => typeof item[1] === 'string')
              .map(([url, name]) => [url.toLowerCase(), name])
          )
        : {},
    disabledWorkerUrls: Array.isArray(entry.disabledWorkerUrls)
      ? entry.disabledWorkerUrls.filter((url): url is string => typeof url === 'string').map((url) => url.toLowerCase())
      : [],
    autoTransferMissingModels: entry.autoTransferMissingModels !== false,
    keepRemoteCopies: entry.keepRemoteCopies === true,
    modelTransferHost: typeof entry.modelTransferHost === 'string' ? entry.modelTransferHost : '',
  };
};

export const remoteWorkersStore = createExternalStore<RemoteWorkersSettings>(makeDefaultSettings());

let loadedEpoch: number | null = null;
let pendingBeforeLoad: Partial<RemoteWorkersSettings> = {};
let pendingSave: { owner: AccountScope; revision: number; settings: RemoteWorkersSettings } | undefined;
let saveTimer: ReturnType<typeof globalThis.setTimeout> | undefined;
let settingsRevision = 0;

const cancelPendingSave = (): void => {
  if (saveTimer !== undefined) {
    globalThis.clearTimeout(saveTimer);
  }
  saveTimer = undefined;
  pendingSave = undefined;
};

const scheduleSave = (settings: RemoteWorkersSettings): void => {
  const owner = captureAccountScope();
  if (!owner.accountId) {
    return;
  }

  settingsRevision += 1;
  const revision = settingsRevision;
  pendingSave = { owner, revision, settings };
  if (saveTimer !== undefined) {
    globalThis.clearTimeout(saveTimer);
  }
  saveTimer = globalThis.setTimeout(() => {
    saveTimer = undefined;
    const pending = pendingSave;
    pendingSave = undefined;
    if (!pending || !isAccountScopeCurrent(pending.owner)) {
      return;
    }
    void apiFetchJson<RemoteWorkersSettings>('/api/v1/remote_workers/settings', {
      method: 'PUT',
      body: JSON.stringify(pending.settings),
      signal: pending.owner.signal,
    }).catch(() => {
      if (isAccountScopeCurrent(pending.owner) && settingsRevision === pending.revision) {
        void loadRemoteWorkersSettings();
      }
    });
  }, 250);
};

export const loadRemoteWorkersSettings = async (): Promise<void> => {
  const owner = captureAccountScope();
  if (!owner.accountId) {
    remoteWorkersStore.setSnapshot(makeDefaultSettings());
    loadedEpoch = null;
    return;
  }

  const loadRevision = settingsRevision;
  try {
    const saved = await apiFetchJson<RemoteWorkersSettings>('/api/v1/remote_workers/settings', {
      signal: owner.signal,
    });
    if (!isAccountScopeCurrent(owner) || loadRevision !== settingsRevision) {
      return;
    }
    const pending = pendingBeforeLoad;
    pendingBeforeLoad = {};
    loadedEpoch = owner.epoch;
    const next = { ...normalizeSettings(saved), ...pending };
    remoteWorkersStore.setSnapshot(next);
    if (Object.keys(pending).length > 0) {
      scheduleSave(next);
    }
  } catch {
    if (!isAccountScopeCurrent(owner) || loadRevision !== settingsRevision) {
      return;
    }
    const pending = pendingBeforeLoad;
    pendingBeforeLoad = {};
    loadedEpoch = owner.epoch;
    const next = { ...makeDefaultSettings(), ...pending };
    remoteWorkersStore.setSnapshot(next);
    if (Object.keys(pending).length > 0) {
      scheduleSave(next);
    }
  }
};

const resetForAccountChange = (): void => {
  settingsRevision += 1;
  loadedEpoch = null;
  pendingBeforeLoad = {};
  cancelPendingSave();
  remoteWorkersStore.setSnapshot(makeDefaultSettings());
  if (captureAccountScope().accountId) {
    void loadRemoteWorkersSettings();
  }
};

registerAccountOwnedResource({
  name: 'remote-workers-server-settings',
  clear: resetForAccountChange,
});

if (captureAccountScope().accountId) {
  void loadRemoteWorkersSettings();
}

export const getRemoteWorkersSettings = (): RemoteWorkersSettings =>
  captureAccountScope().accountId ? remoteWorkersStore.getSnapshot() : makeDefaultSettings();

export const setRemoteWorkersSettings = (patch: Partial<RemoteWorkersSettings>): void => {
  const owner = captureAccountScope();
  if (!owner.accountId) {
    return;
  }
  const safePatch =
    patch.workerUrls === undefined ? patch : { ...patch, workerUrls: stripEmbeddedUrlCredentials(patch.workerUrls) };
  const next = { ...remoteWorkersStore.getSnapshot(), ...safePatch };
  remoteWorkersStore.setSnapshot(next);

  if (loadedEpoch !== owner.epoch) {
    pendingBeforeLoad = { ...pendingBeforeLoad, ...safePatch };
    return;
  }
  scheduleSave(next);
};

/** Worker selection is per account; changing a slot or worker address never changes which worker is paused. */
export const isRemoteWorkerEnabled = (url: string): boolean =>
  !getRemoteWorkersSettings().disabledWorkerUrls.includes(url.toLowerCase());

export const setRemoteWorkerEnabled = (url: string, enabled: boolean): void => {
  const key = url.toLowerCase();
  const disabled = getRemoteWorkersSettings().disabledWorkerUrls;
  if (disabled.includes(key) === !enabled) {
    return;
  }
  setRemoteWorkersSettings({
    disabledWorkerUrls: enabled ? disabled.filter((value) => value !== key) : [...disabled, key],
  });
};

export const getRemoteWorkerName = (url: string, index: number): string => {
  const saved = getRemoteWorkersSettings().workerNames[url.toLowerCase()]?.trim();
  return saved || `Remote ${index + 1}`;
};

export const setRemoteWorkerName = (url: string, name: string): void => {
  const key = url.toLowerCase();
  const names = { ...getRemoteWorkersSettings().workerNames };
  const trimmed = name.trim();
  if (trimmed) {
    names[key] = name;
  } else {
    delete names[key];
  }
  setRemoteWorkersSettings({ workerNames: names });
};

/** Whitespace/newlines/commas/semicolons separate remotes; order determines slot. */
export const getRemoteWorkerUrls = (raw: string): string[] => {
  const seen = new Set<string>();
  const urls: string[] = [];
  for (const part of raw.split(/[;,\r\n\s]+/)) {
    const candidate = part.trim().replace(/\/+$/, '');
    if (!candidate) {
      continue;
    }
    let url: URL;
    try {
      url = new URL(candidate);
    } catch {
      continue;
    }
    if (!['http:', 'https:'].includes(url.protocol) || !url.hostname || url.username || url.password) {
      continue;
    }
    const key = candidate.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      urls.push(candidate);
    }
  }
  return urls;
};
