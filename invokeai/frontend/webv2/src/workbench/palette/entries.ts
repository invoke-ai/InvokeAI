import type { DateRange } from '@platform/search/dateTokens';
import type { CustomHotkeys, HotkeyDefinition } from '@workbench/hotkeys/types';
import type { SettingsSection } from '@workbench/settings/catalog';
import type { SettingsDestination, SettingsSectionId, WorkbenchPreferences } from '@workbench/settings/contracts';
import type { TFunction } from 'i18next';

import { resolveSettingsText } from '@platform/ui/settings/contracts';
import { preferenceSettingsFields } from '@workbench/settings/applicationContributions';
import { settingsCatalog } from '@workbench/settings/catalog';
import fuzzysort from 'fuzzysort';

import { getPaletteContributionKey } from './contributionKey';

/**
 * Normalize sources to PaletteEntry; rank fuzzily within fixed sections so command/settings placement remains
 * stable.
 */

export interface PaletteStageOption {
  id: string;
  label: string;
  isCurrent: boolean;
  apply: () => unknown;
}

/** An inline value-picker pushed over the list (enum settings). */
export interface PaletteStage {
  title: string;
  options: PaletteStageOption[];
  /** Transient preview of the highlighted option (must not persist anything). */
  preview?: (optionId: string) => void;
  /** Revert any active preview; called on every stage exit path. */
  clearPreview?: () => void;
}

export interface PaletteEntry {
  id: string;
  title: string;
  /** Section header the entry renders under; ordered via PALETTE_GROUP_ORDER. */
  group: string;
  /** Localized section header; group remains a stable internal ordering key. */
  groupLabel?: string;
  /** Trailing muted text: a setting's current value, an entity's metadata. */
  subtitle?: string;
  /** Currently applied stage option; renders a check instead of the subtitle. */
  isCurrent?: boolean;
  /** Leading 28×28 thumbnail (image results). */
  thumbnailUrl?: string;
  /** Extra match terms, never rendered. */
  keywords?: string;
  /** Platform-formatted key chips for the bound hotkey, e.g. ['cmd', 'k']. */
  keys?: string[];
  /** Keep the palette open after running (settings toggles). */
  keepOpen?: boolean;
  /** Show in the empty-query launcher state (navigation entries). */
  showInEmptyState?: boolean;
  /** Durable entry whose stable id may be persisted in palette recents. */
  isPersistentRecent: boolean;
  /** Enter pushes this value-picker stage instead of running the entry. */
  stage?: PaletteStage;
  /** mod+Enter alternative, advertised in the footer while highlighted. */
  secondary?: { label: string; run: () => unknown };
  run: () => unknown;
}

/** The structured query providers receive: text with date tokens stripped. */
export interface PaletteProviderQuery {
  text: string;
  /** Resolved created-at bounds; only range-capable providers apply it. */
  range?: DateRange;
}

/** A provider throws this when it cannot search for a reason worth telling the user; its message is shown. */
export class PaletteSearchUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'PaletteSearchUnavailableError';
  }
}

export interface PaletteSearchContext {
  signal: AbortSignal;
}

/** An async entity source the palette can query (workflows, boards, …). */
export interface PaletteSearchProvider {
  /** Globally unique, stable identity (including extension source identity). */
  providerKey: string;
  /** Immutable snapshot of every host value that can affect search results. */
  contextKey: string;
  label: string;
  /**
   * Provider filters by created-at date range. Date-less providers are
   * skipped entirely for pure-date queries (empty text + range), where they
   * could only return broad unfiltered results.
   */
  supportsCreatedAtRange?: boolean;
  /** Searches only once the user scopes into it; root mode offers just its scope row. For costly searches. */
  scopedOnly?: boolean;
  search: (query: PaletteProviderQuery, context: PaletteSearchContext) => Promise<PaletteEntry[]> | PaletteEntry[];
}

export type PaletteRow =
  | { kind: 'label'; id: string; label: string }
  | { kind: 'entry'; id: string; entry: PaletteEntry; matchIndexes?: readonly number[] }
  | { kind: 'provider-error'; id: string; providerKey: string; label: string; retry: () => void }
  | { kind: 'scope'; id: string; providerKey: string; label: string };

export interface ActivePaletteRow {
  index: number;
  row: Exclude<PaletteRow, { kind: 'label' }>;
}

export const getNavigablePaletteRows = (rows: readonly PaletteRow[]): ActivePaletteRow[] =>
  rows.flatMap((row, index): ActivePaletteRow[] => (row.kind === 'label' ? [] : [{ index, row }]));

/** Resolve selection by stable row id, falling back to the first actionable row. */
export const resolveActivePaletteRow = (
  navigableRows: readonly ActivePaletteRow[],
  activeRowId: string | null
): ActivePaletteRow | undefined =>
  (activeRowId ? navigableRows.find((candidate) => candidate.row.id === activeRowId) : undefined) ?? navigableRows[0];

/** Max rows an entity section shows in root mode; depth lives behind the scope. */
export const PROVIDER_ROOT_RESULT_CAP = 3;
/** Entity providers only fire at this query length. */
export const PROVIDER_MIN_QUERY_LENGTH = 2;
/** Group for the synthesized "Search images…"-style scope commands. */
export const SEARCH_SCOPE_GROUP = 'Search in';

const PALETTE_GROUP_ORDER = [
  'Recent',
  'Generation',
  'Queue',
  'Navigation',
  'Layout',
  'App',
  'Canvas',
  'Gallery',
  'Viewer',
  'Workflows',
  'Commands',
  'Settings',
  SEARCH_SCOPE_GROUP,
];

const RECENT_GROUP = 'Recent';
const EMPTY_STATE_RECENT_LIMIT = 6;

/** Matches below this fuzzysort score (0–1) are noise, not results. */
const MATCH_THRESHOLD = 0.2;

const groupRank = (group: string): number => {
  const index = PALETTE_GROUP_ORDER.indexOf(group);

  return index === -1 ? PALETTE_GROUP_ORDER.length : index;
};

/** Omit navigation, caret nudges, and the palette toggle from searchable rows. */
const PALETTE_HIDDEN_COMMANDS = new Set([
  'app.openCommandPalette',
  // Both palettes already ship `buildOpenSettingsEntry` under this id, and it
  // carries search keywords and an empty-state slot the catalog entry lacks.
  'app.openSettings',
  'app.promptWeightDown',
  'app.promptWeightUp',
  'canvas.nextEntity',
  'canvas.prevEntity',
  'gallery.clearSelection',
  'gallery.extendSelectionDown',
  'gallery.extendSelectionLeft',
  'gallery.extendSelectionRight',
  'gallery.extendSelectionUp',
  'gallery.galleryNavDown',
  'gallery.galleryNavDownAlt',
  'gallery.galleryNavLeft',
  'gallery.galleryNavLeftAlt',
  'gallery.galleryNavRight',
  'gallery.galleryNavRightAlt',
  'gallery.galleryNavUp',
  'gallery.galleryNavUpAlt',
  // Focus moves and the toggle act on the focused thumbnail, which a palette open over the gallery has taken.
  'gallery.moveFocusDown',
  'gallery.moveFocusLeft',
  'gallery.moveFocusRight',
  'gallery.moveFocusUp',
  'gallery.toggleFocusedInSelection',
  'viewer.nextItem',
  'viewer.previousItem',
]);

/** App-category commands regrouped into palette-facing sections. */
const APP_COMMAND_GROUPS: Record<string, string> = {
  'app.cancelQueueItem': 'Queue',
  'app.clearQueue': 'Queue',
  'app.focusPrompt': 'Generation',
  'app.invoke': 'Generation',
  'app.invokeFront': 'Generation',
  'app.promptHistoryNext': 'Generation',
  'app.promptHistoryPrev': 'Generation',
  'app.resetPanelLayout': 'Layout',
  'app.selectCanvasTab': 'Navigation',
  'app.selectGenerateTab': 'Navigation',
  'app.selectModelsTab': 'Navigation',
  'app.selectQueueTab': 'Navigation',
  'app.selectUpscalingTab': 'Navigation',
  'app.selectWorkflowsTab': 'Navigation',
  'app.toggleLeftPanel': 'Layout',
  'app.togglePanels': 'Layout',
  'app.togglePreview': 'Layout',
  'app.toggleRightPanel': 'Layout',
};

const TITLE_OVERRIDES: Record<string, string> = {
  'app.invokeFront': 'Invoke (Front of Queue)',
  'app.promptHistoryNext': 'Next Prompt from History',
  'app.promptHistoryPrev': 'Previous Prompt from History',
  'app.selectCanvasTab': 'Go to Canvas',
  'app.selectGenerateTab': 'Go to Generate',
  'app.selectModelsTab': 'Go to Models',
  'app.selectQueueTab': 'Go to Queue',
  'app.selectUpscalingTab': 'Go to Upscaling',
  'app.selectWorkflowsTab': 'Go to Workflows',
};

/** Widget-scoped catalog categories: shown only while the widget type is in the layout. */
const WIDGET_CATEGORY_GROUPS: Record<string, { group: string; typeId: string }> = {
  canvas: { group: 'Canvas', typeId: 'canvas' },
  gallery: { group: 'Gallery', typeId: 'gallery' },
  viewer: { group: 'Viewer', typeId: 'preview' },
  workflows: { group: 'Workflows', typeId: 'workflow' },
};

export const getEntryKeys = (
  definition: HotkeyDefinition | undefined,
  customHotkeys: CustomHotkeys,
  formatHotkey: (key: string) => string[]
): string[] | undefined => {
  if (!definition) {
    return undefined;
  }

  const keys = customHotkeys[definition.id] ?? definition.defaultKeys;

  return keys[0] ? formatHotkey(keys[0]) : undefined;
};

export const buildCatalogCommandEntries = ({
  catalog,
  customHotkeys,
  execute,
  formatHotkey,
  presentWidgetTypeIds,
  t,
  titleOverrides,
}: {
  catalog: readonly HotkeyDefinition[];
  customHotkeys: CustomHotkeys;
  execute: (commandId: string) => unknown;
  formatHotkey: (key: string) => string[];
  presentWidgetTypeIds: ReadonlySet<string>;
  t: TFunction;
  titleOverrides?: Readonly<Record<string, string>>;
}): PaletteEntry[] =>
  catalog
    .filter((definition) => definition.implemented !== false && !PALETTE_HIDDEN_COMMANDS.has(definition.commandId))
    .filter((definition) => {
      const widgetGroup = WIDGET_CATEGORY_GROUPS[definition.category];

      return !widgetGroup || presentWidgetTypeIds.has(widgetGroup.typeId);
    })
    .map((definition) => {
      const widgetGroup = WIDGET_CATEGORY_GROUPS[definition.category];
      const rawGroup = widgetGroup?.group ?? APP_COMMAND_GROUPS[definition.commandId] ?? 'App';
      const groupLabel = t(
        `commandPalette.groups.${rawGroup.replaceAll(' ', '').replace(/^./, (char) => char.toLowerCase())}`,
        { defaultValue: rawGroup }
      );

      return {
        group: rawGroup,
        groupLabel,
        id: definition.commandId,
        isPersistentRecent: true,
        keys: getEntryKeys(definition, customHotkeys, formatHotkey),
        keywords: widgetGroup?.group,
        run: () => execute(definition.commandId),
        showInEmptyState: rawGroup === 'Navigation',
        title:
          titleOverrides?.[definition.commandId] ??
          t(`commandPalette.commands.${definition.commandId.replace('.', '_')}`, {
            defaultValue: TITLE_OVERRIDES[definition.commandId] ?? definition.title,
          }),
      };
    });

export interface SettingsEntryDeps {
  openSettingsSection: (destination: SettingsSectionId | SettingsDestination) => void;
  patchPreferences: (patch: Partial<WorkbenchPreferences>) => unknown;
  /** Transient theme preview while the Theme stage is open (no persistence). */
  previewTheme?: (themeId: string) => void;
  /** Revert a theme preview to the stored preference. */
  clearThemePreview?: () => void;
  languageOptions: ReadonlyArray<{ label: string; value: WorkbenchPreferences['language'] }>;
  themes: ReadonlyArray<{ id: WorkbenchPreferences['themeId']; label: string }>;
}

/** Share Open Settings across hosts; only the editor supplies its resolved hotkey hint. */
export const buildOpenSettingsEntry = (t: TFunction, openSettings: () => void, keys?: string[]): PaletteEntry => ({
  group: 'App',
  groupLabel: t('commandPalette.groups.app'),
  id: 'app.openSettings',
  isPersistentRecent: true,
  keys,
  keywords: 'preferences options',
  run: openSettings,
  showInEmptyState: true,
  title: t('commandPalette.appEntries.openSettings'),
});

/** `sections` are the settings the host can edit (see `useAvailableSettings`); the whole catalog by default. */
export const buildSettingsEntries = (
  preferences: WorkbenchPreferences,
  deps: SettingsEntryDeps,
  t: TFunction,
  sections: readonly SettingsSection[] = settingsCatalog
): PaletteEntry[] => {
  const directPreferenceIds = new Set(preferenceSettingsFields.map(({ field }) => field.id));
  directPreferenceIds.add('themeId');
  const preferenceEntries = preferenceSettingsFields.map<PaletteEntry>(({ field, sectionId }) => {
    const title = resolveSettingsText(field.label, t);
    const section = settingsCatalog.find((candidate) => candidate.id === sectionId);
    const open = () => deps.openSettingsSection({ entryId: field.id, sectionId });
    const value = preferences[field.id as keyof WorkbenchPreferences];
    const entry: PaletteEntry = {
      group: 'Settings',
      groupLabel: t('commandPalette.groups.settings'),
      id: `setting.${field.id}`,
      isPersistentRecent: true,
      keywords: [
        section ? resolveSettingsText(section.label, t) : '',
        field.description ? resolveSettingsText(field.description, t) : '',
        field.keywords ?? '',
      ].join(' '),
      run: open,
      secondary: { label: t('commandPalette.actions.openInSettings'), run: open },
      title,
    };
    if (field.kind === 'boolean' && typeof value === 'boolean') {
      return {
        ...entry,
        keepOpen: true,
        keywords: `${entry.keywords} toggle`,
        run: () => deps.patchPreferences({ [field.id]: !value }),
        subtitle: value ? t('commandPalette.settings.on') : t('commandPalette.settings.off'),
      };
    }
    if (field.kind === 'select') {
      const options =
        field.id === 'language'
          ? deps.languageOptions
          : field.options.map((option) => ({ ...option, label: resolveSettingsText(option.label, t) }));
      return {
        ...entry,
        keywords: `${entry.keywords} ${options.map((option) => option.label).join(' ')}`,
        stage: {
          options: options.map((option) => ({
            apply: () => deps.patchPreferences({ [field.id]: option.value }),
            id: option.value,
            isCurrent: value === option.value,
            label: option.label,
          })),
          title,
        },
        subtitle: options.find((option) => option.value === value)?.label ?? String(value),
      };
    }
    return entry;
  });

  const themeField = settingsCatalog
    .find((section) => section.id === 'appearance')
    ?.entries.find(({ field }) => field.id === 'themeId')?.field;
  const themeTitle = themeField ? resolveSettingsText(themeField.label, t) : t('commandPalette.settings.values.theme');
  const openTheme = () => deps.openSettingsSection({ entryId: 'themeId', sectionId: 'appearance' });
  const themeEntry: PaletteEntry = {
    group: 'Settings',
    groupLabel: t('commandPalette.groups.settings'),
    id: 'setting.themeId',
    isPersistentRecent: true,
    keywords: `${themeField?.keywords ?? ''} ${themeField?.description ? resolveSettingsText(themeField.description, t) : ''} ${deps.themes.map((theme) => theme.label).join(' ')}`,
    run: openTheme,
    secondary: { label: t('commandPalette.actions.openInSettings'), run: openTheme },
    stage: {
      options: deps.themes.map((theme) => ({
        apply: () => deps.patchPreferences({ themeId: theme.id }),
        id: theme.id,
        isCurrent: theme.id === preferences.themeId,
        label: theme.label,
      })),
      title: themeTitle,
      ...(deps.previewTheme && deps.clearThemePreview
        ? { clearPreview: deps.clearThemePreview, preview: deps.previewTheme }
        : {}),
    },
    subtitle: deps.themes.find((theme) => theme.id === preferences.themeId)?.label ?? preferences.themeId,
    title: themeTitle,
  };

  const sectionEntries = sections.map<PaletteEntry>(({ id, label }) => ({
    group: 'Settings',
    groupLabel: t('commandPalette.groups.settings'),
    id: `settings.section.${id}`,
    isPersistentRecent: true,
    keywords: 'settings preferences open',
    run: () => deps.openSettingsSection(id),
    title: t('commandPalette.settings.settingsSection', { section: resolveSettingsText(label, t) }),
  }));
  const fields = sections.flatMap((section) =>
    section.entries
      .filter(({ field }) => field.scope !== 'preference' || !directPreferenceIds.has(field.id))
      .map<PaletteEntry>(({ field }) => ({
        group: 'Settings',
        groupLabel: t('commandPalette.groups.settings'),
        id: `setting.${section.id}.${field.id}`,
        isPersistentRecent: true,
        keywords: [
          resolveSettingsText(section.label, t),
          field.description ? resolveSettingsText(field.description, t) : '',
          field.group ? resolveSettingsText(field.group, t) : '',
          field.keywords ?? '',
          field.kind === 'select' ? field.options.map((option) => resolveSettingsText(option.label, t)).join(' ') : '',
        ].join(' '),
        run: () => deps.openSettingsSection({ entryId: field.id, sectionId: section.id }),
        subtitle: resolveSettingsText(section.label, t),
        title: resolveSettingsText(field.label, t),
      }))
  );

  return [...preferenceEntries, themeEntry, ...sectionEntries, ...fields];
};

interface RankedEntry {
  entry: PaletteEntry;
  matchIndexes?: readonly number[];
  score: number;
}

const toGroupedRows = (groups: Map<string, RankedEntry[]>): PaletteRow[] => {
  const rows: PaletteRow[] = [];
  const orderedGroups = [...groups.keys()].sort(
    (left, right) => groupRank(left) - groupRank(right) || left.localeCompare(right)
  );

  for (const group of orderedGroups) {
    const ranked = groups.get(group) ?? [];

    if (ranked.length === 0) {
      continue;
    }

    rows.push({ id: `label:${group}`, kind: 'label', label: ranked[0]?.entry.groupLabel ?? group });

    for (const { entry, matchIndexes } of ranked) {
      rows.push({ entry, id: entry.id, kind: 'entry', matchIndexes });
    }
  }

  return rows;
};

const buildEmptyStateRows = (
  entries: readonly PaletteEntry[],
  recentIds: readonly string[],
  recentLabel = RECENT_GROUP
): PaletteRow[] => {
  const byId = new Map(entries.map((entry) => [entry.id, entry]));
  const recent = recentIds
    .map((id) => byId.get(id))
    .filter((entry): entry is PaletteEntry => entry !== undefined)
    .slice(0, EMPTY_STATE_RECENT_LIMIT);
  const rows: PaletteRow[] = [];

  if (recent.length > 0) {
    rows.push({ id: `label:${RECENT_GROUP}`, kind: 'label', label: recentLabel });

    for (const entry of recent) {
      rows.push({ entry, id: `recent:${entry.id}`, kind: 'entry' });
    }
  }

  // Entries already shown under Recent don't repeat in their launcher group.
  const shownRecentIds = new Set(recent.map((entry) => entry.id));
  const launcher = new Map<string, RankedEntry[]>();

  for (const entry of entries) {
    if (!entry.showInEmptyState || shownRecentIds.has(entry.id)) {
      continue;
    }

    launcher.set(entry.group, [...(launcher.get(entry.group) ?? []), { entry, score: 0 }]);
  }

  return [...rows, ...toGroupedRows(launcher)];
};

/**
 * Empty queries show recents/navigation. Otherwise rank title/keywords/down-weighted subtitle within fixed
 * sections, breaking ties by recency then title.
 */
export const searchPaletteRows = (
  entries: readonly PaletteEntry[],
  query: string,
  recentIds: readonly string[],
  {
    commandsOnly = false,
    recentLabel = RECENT_GROUP,
    showAllOnEmpty = false,
  }: { commandsOnly?: boolean; recentLabel?: string; showAllOnEmpty?: boolean } = {}
): PaletteRow[] => {
  const searchable = commandsOnly ? entries.filter((entry) => entry.group !== 'Settings') : entries;
  const trimmed = query.trim();

  if (trimmed.length === 0) {
    // Stage mode lists every option before the user types; the root palette
    // shows the curated launcher instead.
    if (showAllOnEmpty) {
      const all = new Map<string, RankedEntry[]>();

      for (const entry of searchable) {
        all.set(entry.group, [...(all.get(entry.group) ?? []), { entry, score: 0 }]);
      }

      return toGroupedRows(all);
    }

    return buildEmptyStateRows(searchable, recentIds, recentLabel);
  }

  const results = fuzzysort.go(trimmed, searchable, {
    keys: [(entry) => entry.title, (entry) => entry.keywords ?? '', (entry) => entry.subtitle ?? ''],
    // fuzzysort 4 caps results at 10 unless told otherwise; 0 lifts the cap.
    limit: 0,
    scoreFn: (result) => Math.max(result[0]?.score ?? 0, (result[1]?.score ?? 0) * 0.9, (result[2]?.score ?? 0) * 0.5),
    threshold: MATCH_THRESHOLD,
  });
  const recencyRank = new Map(recentIds.map((id, index) => [id, index]));
  const groups = new Map<string, RankedEntry[]>();

  for (const result of results) {
    const entry = result.obj;
    const titleResult = result[0];

    groups.set(entry.group, [
      ...(groups.get(entry.group) ?? []),
      {
        entry,
        matchIndexes: titleResult && titleResult.score > 0 ? titleResult.indexes : undefined,
        score: result.score,
      },
    ]);
  }

  for (const ranked of groups.values()) {
    ranked.sort(
      (left, right) =>
        right.score - left.score ||
        (recencyRank.get(left.entry.id) ?? Number.POSITIVE_INFINITY) -
          (recencyRank.get(right.entry.id) ?? Number.POSITIVE_INFINITY) ||
        left.entry.title.localeCompare(right.entry.title)
    );
  }

  return toGroupedRows(groups);
};

/** Row-id prefix for stage options; the dialog strips it to recover the option id. */
export const STAGE_ENTRY_ID_PREFIX = 'stage:';

/** Turn a value-picker stage into filterable entry rows; picking applies and pops. */
export const buildStageEntries = (stage: PaletteStage, onApplied: () => void, t?: TFunction): PaletteEntry[] =>
  stage.options.map((option) => ({
    group: stage.title,
    id: `${STAGE_ENTRY_ID_PREFIX}${option.id}`,
    isCurrent: option.isCurrent,
    isPersistentRecent: false,
    keepOpen: true,
    run: () => {
      void option.apply();
      onApplied();
    },
    subtitle: option.isCurrent ? (t?.('commandPalette.settings.current') ?? 'Current') : undefined,
    title: option.label,
  }));

export interface ProviderResultSection {
  provider: Pick<PaletteSearchProvider, 'providerKey' | 'label'>;
  entries: PaletteEntry[];
  isError: boolean;
  isFetching: boolean;
  isWaitingForDebounce: boolean;
  retry: () => void;
}

/**
 * Entity sections render below local sections as each provider resolves.
 * Empty non-fetching sections drop silently in root mode; `cap` bounds root
 * mode to a teaser (depth lives behind the scope), null means uncapped.
 */
export const buildProviderSectionRows = (
  sections: readonly ProviderResultSection[],
  cap: number | null = PROVIDER_ROOT_RESULT_CAP,
  t?: TFunction
): PaletteRow[] => {
  const rows: PaletteRow[] = [];

  for (const section of sections) {
    if (cap === null && section.isError) {
      // Scoped mode owns the full-width error and Retry presentation.
      continue;
    }

    const visible = cap === null ? section.entries : section.entries.slice(0, cap);

    if (visible.length === 0 && !section.isFetching && !section.isWaitingForDebounce && !section.isError) {
      continue;
    }

    rows.push({
      id: `label:provider:${section.provider.providerKey}`,
      kind: 'label',
      label:
        section.isFetching || section.isWaitingForDebounce
          ? (t?.('commandPalette.states.searchingSection', { label: section.provider.label }) ??
            `${section.provider.label} — Searching…`)
          : section.provider.label,
    });

    if (section.isError && cap !== null) {
      rows.push({
        id: getPaletteContributionKey('provider-error', section.provider.providerKey),
        kind: 'provider-error',
        label:
          t?.('commandPalette.states.retrySearch', { label: section.provider.label.toLowerCase() }) ??
          `Retry ${section.provider.label.toLowerCase()} search`,
        providerKey: section.provider.providerKey,
        retry: section.retry,
      });
    }

    for (const entry of section.isWaitingForDebounce ? [] : visible) {
      rows.push({
        entry,
        id: getPaletteContributionKey('provider-row', `${section.provider.providerKey}:${entry.id}`),
        kind: 'entry',
      });
    }
  }

  return rows;
};

/** Offer one scoped-search escape per provider; an empty query represents date-only filtering. */
export const buildScopeRows = (
  providers: ReadonlyArray<Pick<PaletteSearchProvider, 'providerKey' | 'label'>>,
  query: string,
  t?: TFunction
): PaletteRow[] => {
  if (providers.length === 0) {
    return [];
  }

  return [
    // Distinct id: the scope-command *group* label ("label:Search in") can
    // render in the same list when a query fuzzy-matches those commands.
    { id: 'label:scope-rows', kind: 'label', label: t?.('commandPalette.search.in') ?? SEARCH_SCOPE_GROUP },
    ...providers.map<PaletteRow>((provider) => ({
      id: getPaletteContributionKey('scope', provider.providerKey),
      kind: 'scope',
      label:
        query.length === 0
          ? (t?.('commandPalette.search.byDate', { label: provider.label.toLowerCase() }) ??
            `Search ${provider.label.toLowerCase()} by date`)
          : (t?.('commandPalette.search.forQuery', { label: provider.label.toLowerCase(), query }) ??
            `Search ${provider.label.toLowerCase()} for “${query}”`),
      providerKey: provider.providerKey,
    })),
  ];
};
