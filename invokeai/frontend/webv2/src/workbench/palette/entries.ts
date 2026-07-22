import type { CustomHotkeys, HotkeyDefinition } from '@workbench/hotkeys/types';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';
import type { SettingsSectionId } from '@workbench/widgetContracts';

import { WORKBENCH_LANGUAGE_OPTIONS } from '@platform/i18n/languages';
import { THEMES, THEMES_BY_ID } from '@theme/themes';
import { firstPartyHotkeyCatalog } from '@workbench/hotkeys/catalog';
import { formatHotkeyForPlatform } from '@workbench/hotkeys/keys';
import fuzzysort from 'fuzzysort';

/**
 * Palette content model: every source (catalog commands, extension palette
 * contributions, settings, navigation) normalizes into `PaletteEntry`, and
 * `searchPaletteRows` turns the aggregate into the rendered row list. Ranking
 * is fuzzy (fuzzysort) *within* a section; section order itself is fixed so
 * the list reads spatially — commands above settings, always.
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
}

export interface PaletteEntry {
  id: string;
  title: string;
  /** Section header the entry renders under; ordered via PALETTE_GROUP_ORDER. */
  group: string;
  /** Trailing muted text: a setting's current value, an entity's metadata. */
  subtitle?: string;
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
  /** Enter pushes this value-picker stage instead of running the entry. */
  stage?: PaletteStage;
  /** mod+Enter alternative, advertised in the footer while highlighted. */
  secondary?: { label: string; run: () => unknown };
  run: () => unknown;
}

/** An async entity source the palette can query (workflows, boards, …). */
export interface PaletteSearchProvider {
  id: string;
  label: string;
  search: (query: string) => Promise<PaletteEntry[]> | PaletteEntry[];
}

export type PaletteRow =
  | { kind: 'label'; id: string; label: string }
  | { kind: 'entry'; id: string; entry: PaletteEntry; matchIndexes?: readonly number[] }
  | { kind: 'scope'; id: string; providerId: string; label: string };

/** Max rows an entity section shows in root mode; depth lives behind the scope. */
export const PROVIDER_ROOT_RESULT_CAP = 3;
/** Entity providers only fire at this query length. */
export const PROVIDER_MIN_QUERY_LENGTH = 2;

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
];

const RECENT_GROUP = 'Recent';
const EMPTY_STATE_RECENT_LIMIT = 6;

/** Matches below this fuzzysort score (0–1) are noise, not results. */
const MATCH_THRESHOLD = 0.2;

const groupRank = (group: string): number => {
  const index = PALETTE_GROUP_ORDER.indexOf(group);

  return index === -1 ? PALETTE_GROUP_ORDER.length : index;
};

// ---------------------------------------------------------------------------
// Commands source (hotkey catalog)
// ---------------------------------------------------------------------------

/**
 * Hotkey-only interactions that would be pure noise as palette rows: arrow
 * navigation, caret-relative nudges, and the palette's own toggle.
 */
const PALETTE_HIDDEN_COMMANDS = new Set([
  'app.openCommandPalette',
  'app.promptWeightDown',
  'app.promptWeightUp',
  'canvas.nextEntity',
  'canvas.prevEntity',
  'gallery.clearSelection',
  'gallery.galleryNavDown',
  'gallery.galleryNavDownAlt',
  'gallery.galleryNavLeft',
  'gallery.galleryNavLeftAlt',
  'gallery.galleryNavRight',
  'gallery.galleryNavRightAlt',
  'gallery.galleryNavUp',
  'gallery.galleryNavUpAlt',
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
  'app.selectWorkflowsTab': 'Navigation',
  'app.toggleLeftPanel': 'Layout',
  'app.togglePanels': 'Layout',
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
  'app.selectWorkflowsTab': 'Go to Workflows',
};

/** Widget-scoped catalog categories: shown only while the widget type is in the layout. */
const WIDGET_CATEGORY_GROUPS: Record<string, { group: string; typeId: string }> = {
  canvas: { group: 'Canvas', typeId: 'canvas' },
  gallery: { group: 'Gallery', typeId: 'gallery' },
  viewer: { group: 'Viewer', typeId: 'preview' },
  workflows: { group: 'Workflows', typeId: 'workflow' },
};

const getEntryKeys = (definition: HotkeyDefinition, customHotkeys: CustomHotkeys): string[] | undefined => {
  const keys = customHotkeys[definition.id] ?? definition.defaultKeys;

  return keys[0] ? formatHotkeyForPlatform(keys[0]) : undefined;
};

export const buildCatalogCommandEntries = ({
  catalog = firstPartyHotkeyCatalog,
  customHotkeys,
  execute,
  presentWidgetTypeIds,
}: {
  catalog?: HotkeyDefinition[];
  customHotkeys: CustomHotkeys;
  execute: (commandId: string) => unknown;
  presentWidgetTypeIds: ReadonlySet<string>;
}): PaletteEntry[] =>
  catalog
    .filter((definition) => definition.implemented !== false && !PALETTE_HIDDEN_COMMANDS.has(definition.commandId))
    .filter((definition) => {
      const widgetGroup = WIDGET_CATEGORY_GROUPS[definition.category];

      return !widgetGroup || presentWidgetTypeIds.has(widgetGroup.typeId);
    })
    .map((definition) => {
      const widgetGroup = WIDGET_CATEGORY_GROUPS[definition.category];
      const group = widgetGroup?.group ?? APP_COMMAND_GROUPS[definition.commandId] ?? 'App';

      return {
        group,
        id: definition.commandId,
        keys: getEntryKeys(definition, customHotkeys),
        keywords: widgetGroup?.group,
        run: () => execute(definition.commandId),
        showInEmptyState: group === 'Navigation',
        title: TITLE_OVERRIDES[definition.commandId] ?? definition.title,
      };
    });

// ---------------------------------------------------------------------------
// Settings source
// ---------------------------------------------------------------------------

type BooleanPreferenceKey = {
  [Key in keyof WorkbenchPreferences]: WorkbenchPreferences[Key] extends boolean ? Key : never;
}[keyof WorkbenchPreferences];

const SETTING_TOGGLES: ReadonlyArray<{ key: BooleanPreferenceKey; keywords?: string; title: string }> = [
  { key: 'reduceMotion', keywords: 'animation appearance', title: 'Reduce Motion' },
  { key: 'confirmImageDeletion', keywords: 'delete safety', title: 'Confirm Image Deletion' },
  { key: 'showFocusRegionHighlight', keywords: 'panel outline', title: 'Show Focus Region Highlight' },
  { key: 'enableInformationalPopovers', keywords: 'help tooltips', title: 'Informational Popovers' },
  { key: 'enableModelDescriptions', keywords: 'model manager', title: 'Model Descriptions' },
  { key: 'developerLogEnabled', keywords: 'debug console', title: 'Developer Logging' },
  { key: 'workflowSnapToGrid', keywords: 'workflow editor nodes', title: 'Snap Workflow Nodes to Grid' },
  { key: 'workflowShowMinimap', keywords: 'workflow editor', title: 'Show Workflow Minimap' },
  { key: 'workflowValidateConnections', keywords: 'workflow editor edges', title: 'Validate Workflow Connections' },
];

const SETTINGS_SECTIONS: ReadonlyArray<{ id: SettingsSectionId; title: string }> = [
  { id: 'appearance', title: 'Appearance' },
  { id: 'behavior', title: 'Behavior' },
  { id: 'hotkeys', title: 'Hotkeys' },
  { id: 'project', title: 'Project' },
  { id: 'queue', title: 'Queue' },
  { id: 'workflow', title: 'Workflow' },
  { id: 'developer', title: 'Developer' },
  { id: 'workspace', title: 'Workspace' },
];

export interface SettingsEntryDeps {
  openSettingsSection: (sectionId: SettingsSectionId) => void;
  patchPreferences: (patch: Partial<WorkbenchPreferences>) => unknown;
}

export const buildSettingsEntries = (preferences: WorkbenchPreferences, deps: SettingsEntryDeps): PaletteEntry[] => {
  const toggles = SETTING_TOGGLES.map<PaletteEntry>(({ key, keywords, title }) => ({
    group: 'Settings',
    id: `setting.${key}`,
    keepOpen: true,
    keywords: keywords ? `${keywords} toggle` : 'toggle',
    run: () => deps.patchPreferences({ [key]: !preferences[key] }),
    subtitle: preferences[key] ? 'On' : 'Off',
    title,
  }));

  // Multi-value preferences push an inline value-picker stage; mod+Enter (or
  // the section deep-link rows) still reaches the full settings dialog.
  const enumEntry = ({
    id,
    keywords,
    options,
    sectionId,
    subtitle,
    title,
  }: {
    id: string;
    keywords: string;
    options: PaletteStageOption[];
    sectionId: SettingsSectionId;
    subtitle: string;
    title: string;
  }): PaletteEntry => ({
    group: 'Settings',
    id,
    keywords,
    run: () => deps.openSettingsSection(sectionId),
    secondary: { label: 'Open in Settings', run: () => deps.openSettingsSection(sectionId) },
    stage: { options, title },
    subtitle,
    title,
  });

  const enums: PaletteEntry[] = [
    enumEntry({
      id: 'setting.themeId',
      keywords: 'appearance color dark light',
      options: THEMES.map((theme) => ({
        apply: () => deps.patchPreferences({ themeId: theme.id }),
        id: theme.id,
        isCurrent: theme.id === preferences.themeId,
        label: theme.label,
      })),
      sectionId: 'appearance',
      subtitle: THEMES_BY_ID[preferences.themeId]?.label ?? preferences.themeId,
      title: 'Theme',
    }),
    enumEntry({
      id: 'setting.language',
      keywords: 'locale translation',
      options: WORKBENCH_LANGUAGE_OPTIONS.map((language) => ({
        apply: () => deps.patchPreferences({ language: language.value }),
        id: language.value,
        isCurrent: language.value === preferences.language,
        label: language.label,
      })),
      sectionId: 'appearance',
      subtitle:
        WORKBENCH_LANGUAGE_OPTIONS.find((language) => language.value === preferences.language)?.label ??
        preferences.language,
      title: 'Language',
    }),
    enumEntry({
      id: 'setting.workflowEdgeStyle',
      keywords: 'workflow editor connections',
      options: (['curved', 'square'] as const).map((style) => ({
        apply: () => deps.patchPreferences({ workflowEdgeStyle: style }),
        id: style,
        isCurrent: style === preferences.workflowEdgeStyle,
        label: style === 'square' ? 'Square' : 'Curved',
      })),
      sectionId: 'workflow',
      subtitle: preferences.workflowEdgeStyle === 'square' ? 'Square' : 'Curved',
      title: 'Workflow Edge Style',
    }),
    enumEntry({
      id: 'setting.queueJobsScope',
      keywords: 'queue filter',
      options: (['active-project', 'all'] as const).map((scope) => ({
        apply: () => deps.patchPreferences({ queueJobsScope: scope }),
        id: scope,
        isCurrent: scope === preferences.queueJobsScope,
        label: scope === 'all' ? 'All Projects' : 'Active Project',
      })),
      sectionId: 'queue',
      subtitle: preferences.queueJobsScope === 'all' ? 'All Projects' : 'Active Project',
      title: 'Queue Jobs Scope',
    }),
  ];

  const sections = SETTINGS_SECTIONS.map<PaletteEntry>(({ id, title }) => ({
    group: 'Settings',
    id: `settings.section.${id}`,
    keywords: 'settings preferences open',
    run: () => deps.openSettingsSection(id),
    title: `Settings: ${title}`,
  }));

  return [...toggles, ...enums, ...sections];
};

// ---------------------------------------------------------------------------
// Search / row assembly
// ---------------------------------------------------------------------------

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

    rows.push({ id: `label:${group}`, kind: 'label', label: group });

    for (const { entry, matchIndexes } of ranked) {
      rows.push({ entry, id: entry.id, kind: 'entry', matchIndexes });
    }
  }

  return rows;
};

const buildEmptyStateRows = (entries: readonly PaletteEntry[], recentIds: readonly string[]): PaletteRow[] => {
  const byId = new Map(entries.map((entry) => [entry.id, entry]));
  const recent = recentIds
    .map((id) => byId.get(id))
    .filter((entry): entry is PaletteEntry => entry !== undefined)
    .slice(0, EMPTY_STATE_RECENT_LIMIT);
  const rows: PaletteRow[] = [];

  if (recent.length > 0) {
    rows.push({ id: `label:${RECENT_GROUP}`, kind: 'label', label: RECENT_GROUP });

    for (const entry of recent) {
      rows.push({ entry, id: `recent:${entry.id}`, kind: 'entry' });
    }
  }

  const launcher = new Map<string, RankedEntry[]>();

  for (const entry of entries) {
    if (!entry.showInEmptyState) {
      continue;
    }

    launcher.set(entry.group, [...(launcher.get(entry.group) ?? []), { entry, score: 0 }]);
  }

  return [...rows, ...toGroupedRows(launcher)];
};

/**
 * The full query pipeline: empty query renders the launcher state (recents +
 * navigation); otherwise fuzzysort ranks every entry over title/keywords/
 * subtitle (subtitle down-weighted so value text like "On" cannot dominate),
 * grouped into the fixed section order, ranked within each section by score,
 * then recency, then title.
 */
export const searchPaletteRows = (
  entries: readonly PaletteEntry[],
  query: string,
  recentIds: readonly string[],
  { commandsOnly = false, showAllOnEmpty = false }: { commandsOnly?: boolean; showAllOnEmpty?: boolean } = {}
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

    return buildEmptyStateRows(searchable, recentIds);
  }

  const results = fuzzysort.go(trimmed, searchable, {
    keys: [(entry) => entry.title, (entry) => entry.keywords ?? '', (entry) => entry.subtitle ?? ''],
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

// ---------------------------------------------------------------------------
// Stage / provider row assembly
// ---------------------------------------------------------------------------

/** Turn a value-picker stage into filterable entry rows; picking applies and pops. */
export const buildStageEntries = (stage: PaletteStage, onApplied: () => void): PaletteEntry[] =>
  stage.options.map((option) => ({
    group: stage.title,
    id: `stage:${option.id}`,
    keepOpen: true,
    run: () => {
      void option.apply();
      onApplied();
    },
    subtitle: option.isCurrent ? 'Current' : undefined,
    title: option.label,
  }));

export interface ProviderResultSection {
  provider: Pick<PaletteSearchProvider, 'id' | 'label'>;
  entries: PaletteEntry[];
  isFetching: boolean;
}

/**
 * Entity sections render below local sections as each provider resolves.
 * Empty non-fetching sections drop silently in root mode; `cap` bounds root
 * mode to a teaser (depth lives behind the scope), null means uncapped.
 */
export const buildProviderSectionRows = (
  sections: readonly ProviderResultSection[],
  cap: number | null = PROVIDER_ROOT_RESULT_CAP
): PaletteRow[] => {
  const rows: PaletteRow[] = [];

  for (const section of sections) {
    const visible = cap === null ? section.entries : section.entries.slice(0, cap);

    if (visible.length === 0 && !section.isFetching) {
      continue;
    }

    rows.push({
      id: `label:provider:${section.provider.id}`,
      kind: 'label',
      label: section.isFetching ? `${section.provider.label} — Searching…` : section.provider.label,
    });

    for (const entry of visible) {
      rows.push({ entry, id: entry.id, kind: 'entry' });
    }
  }

  return rows;
};

/** Trailing escape hatches into scoped mode, one per provider. */
export const buildScopeRows = (
  providers: ReadonlyArray<Pick<PaletteSearchProvider, 'id' | 'label'>>,
  query: string
): PaletteRow[] =>
  providers.map((provider) => ({
    id: `scope:${provider.id}`,
    kind: 'scope',
    label: `Search ${provider.label.toLowerCase()} for “${query}”`,
    providerId: provider.id,
  }));
