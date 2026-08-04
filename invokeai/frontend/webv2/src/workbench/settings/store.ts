import type { DeveloperLogLevel, DeveloperLogNamespace } from '@workbench/diagnostics/contracts';
import type { ProjectSettings, StoredRebalancePreset, WorkbenchPreferences } from '@workbench/settings/contracts';

import { getUserStorageScope } from '@features/identity';
import { normalizeWorkbenchLanguage } from '@platform/i18n/languages';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
  type AccountScope,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { createSingleFlight } from '@platform/state/singleFlight';
import { DEFAULT_THEME_ID, isWorkbenchThemeId } from '@theme/themes';
import { deleteClientStateValue, getClientStateValue, setClientStateValue } from '@workbench/projects/api';
import { fetchSessionBlob } from '@workbench/projects/session';

export { WORKBENCH_LANGUAGES } from '@platform/i18n/languages';

const SETTINGS_BASE_STORAGE_KEY = 'invokeai:v7:webv2:settings';
const LEGACY_WORKBENCH_BASE_STORAGE_KEY = 'invokeai:v7:webv2:workbench';
const SETTINGS_CLIENT_STATE_KEY = 'webv2:workbench-settings';

export const DEVELOPER_LOG_LEVELS: DeveloperLogLevel[] = ['trace', 'debug', 'info', 'warn', 'error', 'fatal'];

export const DEVELOPER_LOG_NAMESPACES: DeveloperLogNamespace[] = [
  'canvas',
  'canvas-workflow-integration',
  'config',
  'dnd',
  'events',
  'gallery',
  'generation',
  'metadata',
  'models',
  'system',
  'queue',
  'workflows',
];

export const DEFAULT_PROJECT_SETTINGS: ProjectSettings = {
  antialiasProgressImages: false,
  preferNumericAttentionStyle: false,
  showPromptSyntaxHighlighting: false,
  showProgressDetails: false,
  showProgressImagesInViewer: true,
  useCpuNoise: true,
};

export const DEFAULT_PREFERENCES: WorkbenchPreferences = {
  autoSwitchInvocationRoute: true,
  confirmImageDeletion: true,
  customHotkeys: {},
  developerLogEnabled: true,
  developerLogLevel: 'warn',
  developerLogNamespaces: ['system', 'queue', 'workflows'],
  developerPerformanceTimingsEnabled: false,
  enableInformationalPopovers: true,
  enableModelDescriptions: true,
  generateSectionsOpen: {},
  krea2RebalancePresets: [],
  language: 'en',
  queueJobsScope: 'all',
  reduceMotion: false,
  showFocusRegionHighlight: true,
  themeId: DEFAULT_THEME_ID,
  workflowEdgeStyle: 'curved',
  workflowShowMinimap: true,
  workflowSnapToGrid: false,
  workflowValidateConnections: true,
};

interface WorkbenchSettingsSnapshot {
  /** Account-scope storage key these preferences were hydrated for; null until a load succeeds. */
  hydratedStorageKey: string | null;
  preferences: WorkbenchPreferences;
  scope: 'global' | 'user';
  status: 'idle' | 'loading' | 'ready' | 'error';
  error?: string;
}

const INITIAL_SETTINGS_SNAPSHOT: WorkbenchSettingsSnapshot = {
  hydratedStorageKey: null,
  preferences: DEFAULT_PREFERENCES,
  scope: 'global',
  status: 'idle',
};
const store = createExternalStore<WorkbenchSettingsSnapshot>(INITIAL_SETTINGS_SNAPSHOT);

interface SettingsMutationQueue {
  latestRevision: number;
  owner: AccountScope;
  tail: Promise<void>;
}

let settingsMutationQueue: SettingsMutationQueue | null = null;

registerAccountOwnedResource({
  clear: () => {
    settingsMutationQueue = null;
    store.setSnapshot(INITIAL_SETTINGS_SNAPSHOT);
  },
  name: 'workbench-settings',
});

const isBrowser = (): boolean => typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';

const getSettingsStorageKey = (storageSuffix = getUserStorageScope()): string =>
  `${SETTINGS_BASE_STORAGE_KEY}${storageSuffix}`;

const getLegacyWorkbenchStorageKey = (storageSuffix = getUserStorageScope()): string =>
  `${LEGACY_WORKBENCH_BASE_STORAGE_KEY}${storageSuffix}`;

const getSettingsScope = (storageSuffix = getUserStorageScope()): WorkbenchSettingsSnapshot['scope'] =>
  storageSuffix ? 'user' : 'global';

const isDeveloperLogLevel = (value: unknown): value is DeveloperLogLevel =>
  typeof value === 'string' && DEVELOPER_LOG_LEVELS.includes(value as DeveloperLogLevel);

const normalizeDeveloperLogNamespaces = (values: unknown): DeveloperLogNamespace[] => {
  if (!Array.isArray(values)) {
    return [...DEFAULT_PREFERENCES.developerLogNamespaces];
  }

  const enabled = new Set(
    values.filter(
      (value): value is DeveloperLogNamespace =>
        typeof value === 'string' && DEVELOPER_LOG_NAMESPACES.includes(value as DeveloperLogNamespace)
    )
  );

  return DEVELOPER_LOG_NAMESPACES.filter((namespace) => enabled.has(namespace));
};

const normalizeCustomHotkeys = (values: unknown): Record<string, string[]> => {
  if (!values || typeof values !== 'object' || Array.isArray(values)) {
    return {};
  }

  return Object.fromEntries(
    Object.entries(values)
      .filter(
        (entry): entry is [string, string[]] =>
          typeof entry[0] === 'string' &&
          Array.isArray(entry[1]) &&
          entry[1].every((hotkey) => typeof hotkey === 'string')
      )
      .map(([id, hotkeys]) => [id, hotkeys])
  );
};

const normalizeGenerateSectionsOpen = (values: unknown): Record<string, boolean> => {
  if (!values || typeof values !== 'object' || Array.isArray(values)) {
    return {};
  }

  return Object.fromEntries(
    Object.entries(values).filter(
      (entry): entry is [string, boolean] => typeof entry[0] === 'string' && typeof entry[1] === 'boolean'
    )
  );
};

/**
 * Shape-only: settings keeps the saved records intact and readable, and the generation
 * feature re-validates `weights` against the backend's grammar when it reads them
 * (`normalizeRebalancePresets`). Parsing here would mean importing that feature into the
 * launchpad bundle for no gain.
 */
const normalizeRebalancePresets = (values: unknown): StoredRebalancePreset[] => {
  if (!Array.isArray(values)) {
    return [];
  }

  const seen = new Set<string>();
  const presets: StoredRebalancePreset[] = [];

  for (const entry of values) {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
      continue;
    }

    const { id, label, multiplier, weights } = entry as Partial<StoredRebalancePreset>;

    if (
      typeof id !== 'string' ||
      id.trim() === '' ||
      seen.has(id) ||
      typeof label !== 'string' ||
      label.trim() === '' ||
      typeof weights !== 'string' ||
      typeof multiplier !== 'number' ||
      !Number.isFinite(multiplier)
    ) {
      continue;
    }

    seen.add(id);
    presets.push({ id, label: label.trim(), multiplier, weights });
  }

  return presets;
};

export const normalizeProjectSettings = (settings?: Partial<ProjectSettings>): ProjectSettings => ({
  antialiasProgressImages:
    typeof settings?.antialiasProgressImages === 'boolean'
      ? settings.antialiasProgressImages
      : DEFAULT_PROJECT_SETTINGS.antialiasProgressImages,
  preferNumericAttentionStyle:
    typeof settings?.preferNumericAttentionStyle === 'boolean'
      ? settings.preferNumericAttentionStyle
      : DEFAULT_PROJECT_SETTINGS.preferNumericAttentionStyle,
  showPromptSyntaxHighlighting:
    typeof settings?.showPromptSyntaxHighlighting === 'boolean'
      ? settings.showPromptSyntaxHighlighting
      : DEFAULT_PROJECT_SETTINGS.showPromptSyntaxHighlighting,
  showProgressDetails:
    typeof settings?.showProgressDetails === 'boolean'
      ? settings.showProgressDetails
      : DEFAULT_PROJECT_SETTINGS.showProgressDetails,
  showProgressImagesInViewer:
    typeof settings?.showProgressImagesInViewer === 'boolean'
      ? settings.showProgressImagesInViewer
      : DEFAULT_PROJECT_SETTINGS.showProgressImagesInViewer,
  useCpuNoise: typeof settings?.useCpuNoise === 'boolean' ? settings.useCpuNoise : DEFAULT_PROJECT_SETTINGS.useCpuNoise,
});

type WorkbenchPreferencesInput = Omit<Partial<WorkbenchPreferences>, 'queueJobsScope' | 'workflowEdgeStyle'> & {
  queueJobsScope?: unknown;
  workflowEdgeStyle?: unknown;
};

export const normalizeWorkbenchPreferences = (preferences?: WorkbenchPreferencesInput): WorkbenchPreferences => ({
  autoSwitchInvocationRoute:
    typeof preferences?.autoSwitchInvocationRoute === 'boolean'
      ? preferences.autoSwitchInvocationRoute
      : DEFAULT_PREFERENCES.autoSwitchInvocationRoute,
  confirmImageDeletion:
    typeof preferences?.confirmImageDeletion === 'boolean'
      ? preferences.confirmImageDeletion
      : DEFAULT_PREFERENCES.confirmImageDeletion,
  customHotkeys: normalizeCustomHotkeys(preferences?.customHotkeys),
  developerLogEnabled:
    typeof preferences?.developerLogEnabled === 'boolean'
      ? preferences.developerLogEnabled
      : DEFAULT_PREFERENCES.developerLogEnabled,
  developerLogLevel: isDeveloperLogLevel(preferences?.developerLogLevel)
    ? preferences.developerLogLevel
    : DEFAULT_PREFERENCES.developerLogLevel,
  developerLogNamespaces: normalizeDeveloperLogNamespaces(preferences?.developerLogNamespaces),
  developerPerformanceTimingsEnabled:
    typeof preferences?.developerPerformanceTimingsEnabled === 'boolean'
      ? preferences.developerPerformanceTimingsEnabled
      : DEFAULT_PREFERENCES.developerPerformanceTimingsEnabled,
  enableInformationalPopovers:
    typeof preferences?.enableInformationalPopovers === 'boolean'
      ? preferences.enableInformationalPopovers
      : DEFAULT_PREFERENCES.enableInformationalPopovers,
  enableModelDescriptions:
    typeof preferences?.enableModelDescriptions === 'boolean'
      ? preferences.enableModelDescriptions
      : DEFAULT_PREFERENCES.enableModelDescriptions,
  generateSectionsOpen: normalizeGenerateSectionsOpen(preferences?.generateSectionsOpen),
  krea2RebalancePresets: normalizeRebalancePresets(preferences?.krea2RebalancePresets),
  language: normalizeWorkbenchLanguage(preferences?.language) ?? DEFAULT_PREFERENCES.language,
  queueJobsScope:
    preferences?.queueJobsScope === 'all-projects'
      ? 'all'
      : preferences?.queueJobsScope === 'active-project' || preferences?.queueJobsScope === 'all'
        ? preferences.queueJobsScope
        : DEFAULT_PREFERENCES.queueJobsScope,
  reduceMotion:
    typeof preferences?.reduceMotion === 'boolean' ? preferences.reduceMotion : DEFAULT_PREFERENCES.reduceMotion,
  showFocusRegionHighlight:
    typeof preferences?.showFocusRegionHighlight === 'boolean'
      ? preferences.showFocusRegionHighlight
      : DEFAULT_PREFERENCES.showFocusRegionHighlight,
  themeId: isWorkbenchThemeId(preferences?.themeId) ? preferences.themeId : DEFAULT_PREFERENCES.themeId,
  workflowEdgeStyle:
    preferences?.workflowEdgeStyle === 'square' || preferences?.workflowEdgeStyle === 'straight'
      ? 'square'
      : preferences?.workflowEdgeStyle === 'curved'
        ? preferences.workflowEdgeStyle
        : DEFAULT_PREFERENCES.workflowEdgeStyle,
  workflowShowMinimap:
    typeof preferences?.workflowShowMinimap === 'boolean'
      ? preferences.workflowShowMinimap
      : DEFAULT_PREFERENCES.workflowShowMinimap,
  workflowSnapToGrid:
    typeof preferences?.workflowSnapToGrid === 'boolean'
      ? preferences.workflowSnapToGrid
      : DEFAULT_PREFERENCES.workflowSnapToGrid,
  workflowValidateConnections:
    typeof preferences?.workflowValidateConnections === 'boolean'
      ? preferences.workflowValidateConnections
      : DEFAULT_PREFERENCES.workflowValidateConnections,
});

const parsePreferences = (raw: string | null): WorkbenchPreferences | null => {
  if (!raw) {
    return null;
  }

  try {
    return normalizeWorkbenchPreferences(JSON.parse(raw) as Partial<WorkbenchPreferences>);
  } catch {
    return null;
  }
};

/**
 * The localStorage payload: preferences plus a dirty marker for edits that
 * never reached the backend. A pending local copy outranks the server at the
 * next load, so going offline cannot silently revert a change.
 */
interface StoredSettings {
  preferences: WorkbenchPreferences;
  pendingPush?: boolean;
}

const readLocalSettings = (storageKey: string): StoredSettings | null => {
  if (!isBrowser()) {
    return null;
  }

  try {
    const parsed = JSON.parse(window.localStorage.getItem(storageKey) ?? 'null') as
      | (Partial<StoredSettings> & Partial<WorkbenchPreferences>)
      | null;

    if (!parsed || typeof parsed !== 'object') {
      return null;
    }

    // Early builds stored the bare preferences object.
    if (!parsed.preferences) {
      return { preferences: normalizeWorkbenchPreferences(parsed) };
    }

    return {
      pendingPush: parsed.pendingPush === true || undefined,
      preferences: normalizeWorkbenchPreferences(parsed.preferences),
    };
  } catch {
    return null;
  }
};

const writeLocalSettings = (storageKey: string, preferences: WorkbenchPreferences, pendingPush?: boolean): void => {
  if (!isBrowser()) {
    return;
  }

  const payload: StoredSettings = pendingPush ? { pendingPush, preferences } : { preferences };

  window.localStorage.setItem(storageKey, JSON.stringify(payload));
};

const removeLocalSettings = (storageKey: string): void => {
  if (!isBrowser()) {
    return;
  }

  window.localStorage.removeItem(storageKey);
};

const readLegacyLocalPreferences = (storageSuffix: string): WorkbenchPreferences | null => {
  if (!isBrowser()) {
    return null;
  }

  try {
    const raw = window.localStorage.getItem(getLegacyWorkbenchStorageKey(storageSuffix));
    const parsed = raw ? (JSON.parse(raw) as { state?: { account?: { preferences?: unknown } } }) : null;

    return parsed?.state?.account?.preferences
      ? normalizeWorkbenchPreferences(parsed.state.account.preferences as Partial<WorkbenchPreferences>)
      : null;
  } catch {
    return null;
  }
};

const loadLegacySessionPreferences = async (owner: AccountScope): Promise<WorkbenchPreferences | null> => {
  const blob = await fetchSessionBlob(owner.signal);

  assertAccountScopeCurrent(owner);
  return blob?.account.preferences ? normalizeWorkbenchPreferences(blob.account.preferences) : null;
};

const replaceSnapshot = (
  preferences: WorkbenchPreferences,
  status: WorkbenchSettingsSnapshot['status'],
  hydratedStorageKey: string | null = store.getSnapshot().hydratedStorageKey,
  storageSuffix = getUserStorageScope()
): void => {
  store.setSnapshot({ hydratedStorageKey, preferences, scope: getSettingsScope(storageSuffix), status });
};

const pushPreferences = (preferences: WorkbenchPreferences, signal: AbortSignal): Promise<void> =>
  setClientStateValue(SETTINGS_CLIENT_STATE_KEY, JSON.stringify(preferences), signal);

const getSettingsMutationQueue = (owner: AccountScope): SettingsMutationQueue => {
  if (settingsMutationQueue?.owner === owner) {
    return settingsMutationQueue;
  }

  settingsMutationQueue = { latestRevision: 0, owner, tail: Promise.resolve() };

  return settingsMutationQueue;
};

const enqueueSettingsMutation = (
  owner: AccountScope,
  operation: (queue: SettingsMutationQueue, revision: number) => Promise<void>
): Promise<void> => {
  const queue = getSettingsMutationQueue(owner);
  const revision = (queue.latestRevision += 1);
  const run = (): Promise<void> => operation(queue, revision);
  const result = queue.tail.then(run, run);

  queue.tail = result.catch(() => undefined);

  return result;
};

const resolveSettings = async (
  local: StoredSettings | null,
  owner: AccountScope,
  storageSuffix: string
): Promise<WorkbenchPreferences> => {
  if (local?.pendingPush) {
    await pushPreferences(local.preferences, owner.signal);
    assertAccountScopeCurrent(owner);

    return local.preferences;
  }

  const backendPreferences = parsePreferences(await getClientStateValue(SETTINGS_CLIENT_STATE_KEY, owner.signal));

  assertAccountScopeCurrent(owner);

  if (backendPreferences) {
    return backendPreferences;
  }

  // First contact for this account: adopt whatever the legacy locations
  // hold and write the new backend key once.
  const preferences =
    (await loadLegacySessionPreferences(owner)) ??
    local?.preferences ??
    readLegacyLocalPreferences(storageSuffix) ??
    normalizeWorkbenchPreferences();

  await pushPreferences(preferences, owner.signal);
  assertAccountScopeCurrent(owner);

  return preferences;
};

const settingsLoad = createSingleFlight<WorkbenchPreferences>();

/**
 * Resolve settings once per account scope (a sign-in/out switches scope and
 * reloads); later calls return the snapshot. Offline, the local copy serves
 * and the next load retries the backend.
 */
export const loadWorkbenchSettings = (): Promise<WorkbenchPreferences> => {
  const owner = captureAccountScope();
  const storageSuffix = owner.accountId === null ? getUserStorageScope() : owner.storageSuffix;
  const storageKey = getSettingsStorageKey(storageSuffix);

  if (store.getSnapshot().hydratedStorageKey === storageKey) {
    return Promise.resolve(getWorkbenchPreferences());
  }

  return settingsLoad.run(`${storageKey}:${owner.epoch}`, async () => {
    store.patchSnapshot({ scope: getSettingsScope(storageSuffix), status: 'loading' });
    const local = readLocalSettings(storageKey);

    try {
      const preferences = await resolveSettings(local, owner, storageSuffix);

      assertAccountScopeCurrent(owner);
      writeLocalSettings(storageKey, preferences);
      replaceSnapshot(preferences, 'ready', storageKey, storageSuffix);

      return preferences;
    } catch (error) {
      if (!isAccountScopeCurrent(owner)) {
        throw error;
      }

      const fallback = local?.preferences ?? readLegacyLocalPreferences(storageSuffix);
      const preferences = fallback ?? normalizeWorkbenchPreferences();

      writeLocalSettings(storageKey, preferences, local?.pendingPush);
      store.setSnapshot({
        error: error instanceof Error ? error.message : 'Failed to load settings from the backend.',
        hydratedStorageKey: null,
        preferences,
        scope: getSettingsScope(storageSuffix),
        status: fallback ? 'ready' : 'error',
      });

      return preferences;
    }
  });
};

export const useWorkbenchSettings = (): WorkbenchSettingsSnapshot => store.useSnapshot();

export const useWorkbenchPreferences = (): WorkbenchPreferences =>
  store.useSelector((snapshot) => snapshot.preferences);

type EqualityFn<T> = (left: T, right: T) => boolean;

export const useWorkbenchSettingsSelector = <Selected>(
  selector: (snapshot: WorkbenchSettingsSnapshot) => Selected,
  isEqual: EqualityFn<Selected> = Object.is
): Selected => store.useSelector(selector, isEqual);

export const useWorkbenchPreferenceSelector = <Selected>(
  selector: (preferences: WorkbenchPreferences) => Selected,
  isEqual?: EqualityFn<Selected>
): Selected => useWorkbenchSettingsSelector((snapshot) => selector(snapshot.preferences), isEqual);

export const getWorkbenchPreferences = (): WorkbenchPreferences => store.getSnapshot().preferences;

/** Non-React runtimes use this to keep infrastructure configuration current. */
export const subscribeWorkbenchPreferences = (listener: (preferences: WorkbenchPreferences) => void): (() => void) => {
  listener(store.getSnapshot().preferences);

  return store.subscribe(() => listener(store.getSnapshot().preferences));
};

export const getWorkbenchReduceMotion = (): boolean => store.getSnapshot().preferences.reduceMotion;

export const useWorkbenchReduceMotion = (): boolean =>
  store.useSelector((snapshot) => snapshot.preferences.reduceMotion);

export const patchWorkbenchPreferences = async (preferences: Partial<WorkbenchPreferences>): Promise<void> => {
  const owner = captureAccountScope();
  const storageSuffix = owner.accountId === null ? getUserStorageScope() : owner.storageSuffix;
  const storageKey = getSettingsStorageKey(storageSuffix);
  const next = normalizeWorkbenchPreferences({ ...store.getSnapshot().preferences, ...preferences });

  replaceSnapshot(next, 'ready', store.getSnapshot().hydratedStorageKey, storageSuffix);
  writeLocalSettings(storageKey, next, true);

  await enqueueSettingsMutation(owner, async (queue, revision) => {
    if (!isAccountScopeCurrent(owner)) {
      return;
    }

    try {
      await pushPreferences(next, owner.signal);

      if (!isAccountScopeCurrent(owner)) {
        return;
      }
      if (queue.latestRevision === revision) {
        writeLocalSettings(storageKey, next);
      }
    } catch (error) {
      if (!isAccountScopeCurrent(owner) || queue.latestRevision !== revision) {
        return;
      }

      store.patchSnapshot({
        error: error instanceof Error ? error.message : 'Failed to save settings.',
        status: 'error',
      });
    }
  });
};

export const clearWorkbenchSettings = async (): Promise<void> => {
  const owner = captureAccountScope();
  const storageSuffix = owner.accountId === null ? getUserStorageScope() : owner.storageSuffix;
  const storageKey = getSettingsStorageKey(storageSuffix);

  removeLocalSettings(storageKey);
  replaceSnapshot(normalizeWorkbenchPreferences(), 'ready', storageKey, storageSuffix);

  await enqueueSettingsMutation(owner, async () => {
    if (!isAccountScopeCurrent(owner)) {
      return;
    }

    try {
      await deleteClientStateValue(SETTINGS_CLIENT_STATE_KEY, owner.signal);
    } catch {
      // Backend persistence is best-effort; the local reset should still complete.
    }
  });
};
