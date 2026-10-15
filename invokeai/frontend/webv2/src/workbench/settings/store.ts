import type { LogLevel, LogNamespace } from '@platform/logging/contracts';
import type {
  ProjectSettings,
  ProjectSortId,
  ProjectsViewId,
  StoredGeneratePreset,
  StoredRebalancePreset,
  WorkbenchPreferences,
} from '@workbench/settings/contracts';

import { getUserStorageScope } from '@features/identity';
import { normalizeWorkbenchLanguage } from '@platform/i18n/languages';
import { isLogLevel, isLogNamespace, LOG_LEVELS, LOG_NAMESPACES } from '@platform/logging/contracts';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
  type AccountScope,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { createSingleFlight } from '@platform/state/singleFlight';
import { isPromptFontSize } from '@theme/scale';
import { DEFAULT_THEME_ID, resolveWorkbenchThemeId } from '@theme/themes';
import { deleteClientStateValue, getClientStateValue, setClientStateValue } from '@workbench/projects/api';
import { fetchSessionBlob } from '@workbench/projects/session';

export { WORKBENCH_LANGUAGES } from '@platform/i18n/languages';

const SETTINGS_BASE_STORAGE_KEY = 'invokeai:v7:webv2:settings';
const LEGACY_WORKBENCH_BASE_STORAGE_KEY = 'invokeai:v7:webv2:workbench';
const SETTINGS_CLIENT_STATE_KEY = 'webv2:workbench-settings';

export const DEVELOPER_LOG_LEVELS: readonly LogLevel[] = LOG_LEVELS;

export const DEVELOPER_LOG_NAMESPACES: readonly LogNamespace[] = LOG_NAMESPACES;

export const DEFAULT_PROJECT_SETTINGS: ProjectSettings = {
  antialiasProgressImages: false,
  showProgressImagesInViewer: true,
  useCpuNoise: true,
};

export const DEFAULT_PREFERENCES: WorkbenchPreferences = {
  alphaNoticeAcknowledged: false,
  autoSwitchInvocationRoute: true,
  confirmImageDeletion: true,
  customHotkeys: {},
  developerConsoleOutputEnabled: false,
  developerLogEnabled: true,
  developerLogLevel: 'warn',
  developerLogNamespaces: [...LOG_NAMESPACES],
  developerPerformanceTimingsEnabled: false,
  enableInformationalPopovers: true,
  enableModelDescriptions: true,
  generatePresets: [],
  generateSectionsOpen: {},
  highContrast: false,
  krea2RebalancePresets: [],
  language: 'en',
  launchpadPinnedProjectIds: [],
  launchpadProjectsSort: 'edited',
  launchpadProjectsView: 'grid',
  notifyOnEnqueue: true,
  preferNumericAttentionStyle: false,
  promptFontSize: 'default',
  queueJobsScope: 'all',
  reduceMotion: false,
  showPromptSyntaxHighlighting: true,
  showFocusRegionHighlight: true,
  themeId: DEFAULT_THEME_ID,
  whatsNewSeenVersion: null,
  workflowEdgeStyle: 'curved',
  workflowEdgesBehindNodes: false,
  workflowGroupNodesByCategory: true,
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

/** The pre-2026-09 default; installs that merely persisted it never chose a narrowed selection. */
const LEGACY_DEFAULT_LOG_NAMESPACES: readonly LogNamespace[] = ['queue', 'system', 'workflows'];

/** A user-narrowed selection stays narrowed, including namespaces added after it was saved; reset restores defaults. */
const normalizeDeveloperLogNamespaces = (values: unknown): LogNamespace[] => {
  if (!Array.isArray(values)) {
    return [...DEFAULT_PREFERENCES.developerLogNamespaces];
  }

  const enabled = new Set(values.filter(isLogNamespace));
  const selected = LOG_NAMESPACES.filter((namespace) => enabled.has(namespace));
  const isLegacyDefault =
    selected.length === LEGACY_DEFAULT_LOG_NAMESPACES.length &&
    selected.every((namespace, index) => namespace === LEGACY_DEFAULT_LOG_NAMESPACES[index]);

  return isLegacyDefault ? [...DEFAULT_PREFERENCES.developerLogNamespaces] : selected;
};

/** Keep view guards local so shared settings do not pull Launchpad code into the editor bundle. */
const isProjectsViewId = (value: unknown): value is ProjectsViewId => value === 'grid' || value === 'list';

const isProjectSortId = (value: unknown): value is ProjectSortId =>
  value === 'edited' || value === 'created' || value === 'name';

const normalizePinnedProjectIds = (values: unknown): string[] =>
  Array.isArray(values) ? [...new Set(values.filter((value): value is string => typeof value === 'string'))] : [];

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
 * Validate record shape here; generation validates weight grammar when reading, preserving Launchpad's bundle
 * boundary.
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

/** Shape-only, like {@link normalizeRebalancePresets}: the generation feature re-normalizes `values` on apply. */
const normalizeGeneratePresets = (values: unknown): StoredGeneratePreset[] => {
  if (!Array.isArray(values)) {
    return [];
  }

  const seen = new Set<string>();
  const presets: StoredGeneratePreset[] = [];

  for (const entry of values) {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
      continue;
    }

    const { id, label, values: snapshot } = entry as Partial<StoredGeneratePreset>;

    if (
      typeof id !== 'string' ||
      id.trim() === '' ||
      seen.has(id) ||
      typeof label !== 'string' ||
      label.trim() === '' ||
      !snapshot ||
      typeof snapshot !== 'object' ||
      Array.isArray(snapshot)
    ) {
      continue;
    }

    seen.add(id);
    presets.push({ id, label: label.trim(), values: snapshot });
  }

  return presets;
};

export const normalizeProjectSettings = (settings?: Partial<ProjectSettings>): ProjectSettings => ({
  antialiasProgressImages:
    typeof settings?.antialiasProgressImages === 'boolean'
      ? settings.antialiasProgressImages
      : DEFAULT_PROJECT_SETTINGS.antialiasProgressImages,
  showProgressImagesInViewer:
    typeof settings?.showProgressImagesInViewer === 'boolean'
      ? settings.showProgressImagesInViewer
      : DEFAULT_PROJECT_SETTINGS.showProgressImagesInViewer,
  useCpuNoise: typeof settings?.useCpuNoise === 'boolean' ? settings.useCpuNoise : DEFAULT_PROJECT_SETTINGS.useCpuNoise,
});

type WorkbenchPreferencesInput = Omit<
  Partial<WorkbenchPreferences>,
  'launchpadProjectsSort' | 'launchpadProjectsView' | 'queueJobsScope' | 'workflowEdgeStyle'
> & {
  launchpadProjectsSort?: unknown;
  launchpadProjectsView?: unknown;
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
  developerConsoleOutputEnabled:
    typeof preferences?.developerConsoleOutputEnabled === 'boolean'
      ? preferences.developerConsoleOutputEnabled
      : DEFAULT_PREFERENCES.developerConsoleOutputEnabled,
  developerLogLevel: isLogLevel(preferences?.developerLogLevel)
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
  generatePresets: normalizeGeneratePresets(preferences?.generatePresets),
  generateSectionsOpen: normalizeGenerateSectionsOpen(preferences?.generateSectionsOpen),
  krea2RebalancePresets: normalizeRebalancePresets(preferences?.krea2RebalancePresets),
  language: normalizeWorkbenchLanguage(preferences?.language) ?? DEFAULT_PREFERENCES.language,
  launchpadPinnedProjectIds: normalizePinnedProjectIds(preferences?.launchpadPinnedProjectIds),
  launchpadProjectsSort: isProjectSortId(preferences?.launchpadProjectsSort)
    ? preferences.launchpadProjectsSort
    : DEFAULT_PREFERENCES.launchpadProjectsSort,
  launchpadProjectsView: isProjectsViewId(preferences?.launchpadProjectsView)
    ? preferences.launchpadProjectsView
    : DEFAULT_PREFERENCES.launchpadProjectsView,
  notifyOnEnqueue:
    typeof preferences?.notifyOnEnqueue === 'boolean'
      ? preferences.notifyOnEnqueue
      : DEFAULT_PREFERENCES.notifyOnEnqueue,
  preferNumericAttentionStyle:
    typeof preferences?.preferNumericAttentionStyle === 'boolean'
      ? preferences.preferNumericAttentionStyle
      : DEFAULT_PREFERENCES.preferNumericAttentionStyle,
  promptFontSize: isPromptFontSize(preferences?.promptFontSize)
    ? preferences.promptFontSize
    : DEFAULT_PREFERENCES.promptFontSize,
  queueJobsScope:
    preferences?.queueJobsScope === 'all-projects'
      ? 'all'
      : preferences?.queueJobsScope === 'active-project' || preferences?.queueJobsScope === 'all'
        ? preferences.queueJobsScope
        : DEFAULT_PREFERENCES.queueJobsScope,
  highContrast:
    typeof preferences?.highContrast === 'boolean' ? preferences.highContrast : DEFAULT_PREFERENCES.highContrast,
  alphaNoticeAcknowledged:
    typeof preferences?.alphaNoticeAcknowledged === 'boolean'
      ? preferences.alphaNoticeAcknowledged
      : DEFAULT_PREFERENCES.alphaNoticeAcknowledged,
  reduceMotion:
    typeof preferences?.reduceMotion === 'boolean' ? preferences.reduceMotion : DEFAULT_PREFERENCES.reduceMotion,
  showFocusRegionHighlight:
    typeof preferences?.showFocusRegionHighlight === 'boolean'
      ? preferences.showFocusRegionHighlight
      : DEFAULT_PREFERENCES.showFocusRegionHighlight,
  showPromptSyntaxHighlighting:
    typeof preferences?.showPromptSyntaxHighlighting === 'boolean'
      ? preferences.showPromptSyntaxHighlighting
      : DEFAULT_PREFERENCES.showPromptSyntaxHighlighting,
  themeId: resolveWorkbenchThemeId(preferences?.themeId) ?? DEFAULT_PREFERENCES.themeId,
  whatsNewSeenVersion:
    typeof preferences?.whatsNewSeenVersion === 'string'
      ? preferences.whatsNewSeenVersion
      : DEFAULT_PREFERENCES.whatsNewSeenVersion,
  workflowEdgeStyle:
    preferences?.workflowEdgeStyle === 'square' || preferences?.workflowEdgeStyle === 'straight'
      ? 'square'
      : preferences?.workflowEdgeStyle === 'curved'
        ? preferences.workflowEdgeStyle
        : DEFAULT_PREFERENCES.workflowEdgeStyle,
  workflowEdgesBehindNodes:
    typeof preferences?.workflowEdgesBehindNodes === 'boolean'
      ? preferences.workflowEdgesBehindNodes
      : DEFAULT_PREFERENCES.workflowEdgesBehindNodes,
  workflowGroupNodesByCategory:
    typeof preferences?.workflowGroupNodesByCategory === 'boolean'
      ? preferences.workflowGroupNodesByCategory
      : DEFAULT_PREFERENCES.workflowGroupNodesByCategory,
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
