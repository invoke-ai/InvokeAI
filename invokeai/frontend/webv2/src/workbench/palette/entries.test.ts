import type { TFunction } from 'i18next';

import { firstPartyHotkeyCatalog } from '@workbench/hotkeys/catalog';
import { formatHotkeyForPlatform } from '@workbench/hotkeys/keys';
import { DEFAULT_PREFERENCES } from '@workbench/settings/store';
import { describe, expect, it, vi } from 'vitest';

import type { PaletteEntry } from './entries';

import { getPaletteContributionKey } from './contributionKey';
import {
  buildCatalogCommandEntries,
  buildProviderSectionRows,
  buildScopeRows,
  buildSettingsEntries,
  buildStageEntries,
  buildWidgetEntries,
  SEARCH_SCOPE_GROUP,
  searchPaletteRows,
  getNavigablePaletteRows,
  resolveActivePaletteRow,
} from './entries';

const noop = () => undefined;
const translations: Record<string, string> = {
  'commandPalette.actions.openInSettings': 'Open in Settings',
  'commandPalette.groups.app': 'App',
  'commandPalette.groups.canvas': 'Canvas',
  'topbar.invoke.invokeFront': 'Invoke (front of queue)',
  'commandPalette.groups.generation': 'Generation',
  'commandPalette.groups.navigation': 'Navigation',
  'commandPalette.groups.settings': 'Settings',
  'commandPalette.settings.off': 'Off',
  'commandPalette.settings.on': 'On',
  'commandPalette.settings.settingsSection': 'Settings: {{section}}',
  'commandPalette.settings.values.activeProject': 'Active Project',
  'commandPalette.settings.values.allProjects': 'All Projects',
  'commandPalette.settings.values.curved': 'Curved',
  'commandPalette.settings.values.language': 'Language',
  'commandPalette.settings.values.queueJobsScope': 'Queue Jobs Scope',
  'commandPalette.settings.values.square': 'Square',
  'commandPalette.settings.values.theme': 'Theme',
  'commandPalette.settings.values.workflowEdgeStyle': 'Workflow Edge Style',
};
const t = ((key: string, options?: Record<string, unknown>) => {
  let value =
    translations[key] ?? (typeof options?.defaultValue === 'string' ? options.defaultValue : key.split('.').at(-1));

  for (const [name, replacement] of Object.entries(options ?? {})) {
    value = value?.replace(`{{${name}}}`, String(replacement));
  }

  return value;
}) as TFunction;
const settingsEntryDeps = {
  languageOptions: [{ label: 'English', value: 'en' as const }],
  openSettingsSection: vi.fn(),
  patchPreferences: vi.fn(),
  themes: [{ id: 'classic' as const, label: 'Classic' }],
};

const makeEntry = (overrides: Partial<PaletteEntry> & Pick<PaletteEntry, 'id' | 'title' | 'group'>): PaletteEntry => ({
  isPersistentRecent: true,
  run: noop,
  ...overrides,
});

describe('buildCatalogCommandEntries', () => {
  const build = (presentWidgetTypeIds: ReadonlySet<string>, execute: (commandId: string) => unknown = noop) =>
    buildCatalogCommandEntries({
      catalog: firstPartyHotkeyCatalog,
      customHotkeys: {},
      execute,
      formatHotkey: formatHotkeyForPlatform,
      presentWidgetTypeIds,
      t,
    });

  it('surfaces app commands with palette-facing titles and groups', () => {
    const entries = build(new Set());
    const invoke = entries.find((entry) => entry.id === 'app.invoke');

    expect(invoke).toMatchObject({ group: 'Generation', title: 'Invoke' });
    expect(entries.find((entry) => entry.id === 'app.selectWorkflowsTab')).toMatchObject({
      group: 'Navigation',
      showInEmptyState: true,
      title: 'Open Workflow editor',
    });
  });

  it('offers each widget once, under its open command where it has one', () => {
    const show = vi.fn();
    const commands = build(new Set()).map((entry) => entry.id);
    const widgetEntries = buildWidgetEntries({
      catalog: firstPartyHotkeyCatalog,
      customHotkeys: { 'app.selectGenerateTab': ['g'] },
      formatHotkey: formatHotkeyForPlatform,
      show,
      t,
      widgets: [
        { label: 'Generate', typeId: 'generate' },
        { label: 'Image Map', typeId: 'image-map' },
      ],
    });

    expect(commands).not.toContain('app.selectGenerateTab');
    expect(
      widgetEntries.map(({ group, id, keys, showInEmptyState }) => ({ group, id, keys, showInEmptyState }))
    ).toEqual([
      { group: 'Navigation', id: 'app.selectGenerateTab', keys: ['g'], showInEmptyState: true },
      { group: 'Layout', id: 'widget.show.image-map', keys: undefined, showInEmptyState: false },
    ]);
    widgetEntries[0]!.run();
    expect(show).toHaveBeenCalledWith('generate');
  });

  it('leaves out widget commands nothing handles right now', () => {
    const ids = buildCatalogCommandEntries({
      catalog: firstPartyHotkeyCatalog,
      customHotkeys: {},
      execute: noop,
      formatHotkey: formatHotkeyForPlatform,
      isHandled: (commandId) => commandId !== 'viewer.zoomToFit',
      presentWidgetTypeIds: new Set(['preview']),
      t,
    }).map((entry) => entry.id);

    expect(ids).toContain('viewer.toggleMetadata');
    expect(ids).not.toContain('viewer.zoomToFit');
  });

  it('shows widget-scoped commands only while the widget type is in the layout', () => {
    const withoutCanvas = build(new Set(['gallery'])).map((entry) => entry.id);
    const withCanvas = build(new Set(['canvas'])).map((entry) => entry.id);

    expect(withoutCanvas).not.toContain('canvas.undo');
    expect(withoutCanvas).toContain('gallery.remix');
    expect(withCanvas).toContain('canvas.undo');
  });

  it('formats the bound hotkey (custom overrides win) and executes by command id', () => {
    const execute = vi.fn();
    const entries = buildCatalogCommandEntries({
      catalog: firstPartyHotkeyCatalog,
      customHotkeys: { 'app.clearQueue': ['mod+shift+q'] },
      execute,
      // Pinned so the result does not depend on the machine; keys.test.ts owns the macOS order.
      formatHotkey: (hotkey) => formatHotkeyForPlatform(hotkey, false),
      presentWidgetTypeIds: new Set(),
      t,
    });
    const invoke = entries.find((entry) => entry.id === 'app.invoke');
    const clearQueue = entries.find((entry) => entry.id === 'app.clearQueue');

    expect(invoke?.keys).toEqual(['ctrl', 'enter']);
    expect(clearQueue?.keys).toEqual(['ctrl', 'shift', 'q']);

    invoke?.run();
    expect(execute).toHaveBeenCalledWith('app.invoke');
  });
});

describe('buildSettingsEntries', () => {
  it('renders toggle state and flips the preference in place', () => {
    const patchPreferences = vi.fn();
    const entries = buildSettingsEntries(
      { ...DEFAULT_PREFERENCES, reduceMotion: false },
      { ...settingsEntryDeps, openSettingsSection: vi.fn(), patchPreferences },
      t
    );
    const reduceMotion = entries.find((entry) => entry.id === 'setting.reduceMotion');

    expect(reduceMotion).toMatchObject({ keepOpen: true, subtitle: 'Off' });

    reduceMotion?.run();
    expect(patchPreferences).toHaveBeenCalledWith({ reduceMotion: true });
  });

  it('deep-links enum preferences and sections to the settings dialog', () => {
    const openSettingsSection = vi.fn();
    const entries = buildSettingsEntries(
      DEFAULT_PREFERENCES,
      {
        ...settingsEntryDeps,
        openSettingsSection,
        patchPreferences: vi.fn(),
      },
      t
    );
    const theme = entries.find((entry) => entry.id === 'setting.themeId');
    const workflowSection = entries.find((entry) => entry.id === 'settings.section.workflow');

    expect(theme?.subtitle).toBe('Classic');

    theme?.run();
    expect(openSettingsSection).toHaveBeenCalledWith({ entryId: 'themeId', sectionId: 'appearance' });

    workflowSection?.run();
    expect(openSettingsSection).toHaveBeenCalledWith('workflow');
  });

  it('finds widget fields by catalog aliases and navigates to the exact entry', () => {
    const openSettingsSection = vi.fn();
    const entries = buildSettingsEntries(DEFAULT_PREFERENCES, { ...settingsEntryDeps, openSettingsSection }, t);
    const rows = searchPaletteRows(entries, 'density', []);
    const imageSize = rows.find((row) => row.kind === 'entry' && row.entry.id === 'setting.gallery.imageSize');

    expect(imageSize?.kind).toBe('entry');
    if (imageSize?.kind === 'entry') {
      imageSize.entry.run();
    }
    expect(openSettingsSection).toHaveBeenCalledWith({ entryId: 'imageSize', sectionId: 'gallery' });
    expect(entries.some((entry) => entry.id === 'settings.section.canvas')).toBe(true);
  });

  it('carries a value-picker stage on enum preferences that applies and marks the current value', () => {
    const patchPreferences = vi.fn();
    const entries = buildSettingsEntries(
      { ...DEFAULT_PREFERENCES, workflowEdgeStyle: 'curved' },
      { ...settingsEntryDeps, openSettingsSection: vi.fn(), patchPreferences },
      t
    );
    const edgeStyle = entries.find((entry) => entry.id === 'setting.workflowEdgeStyle');

    expect(edgeStyle?.stage?.options.map((option) => [option.label, option.isCurrent])).toEqual([
      ['Curved', true],
      ['Square', false],
    ]);
    expect(edgeStyle?.secondary?.label).toBe('Open in Settings');

    edgeStyle?.stage?.options[1]?.apply();
    expect(patchPreferences).toHaveBeenCalledWith({ workflowEdgeStyle: 'square' });
  });
});

describe('searchPaletteRows', () => {
  it('returns every matching entry rather than the first ten', () => {
    const entries: PaletteEntry[] = Array.from({ length: 12 }, (_, index) => ({
      group: 'App',
      id: `portrait-${String(index)}`,
      isPersistentRecent: false,
      run: () => undefined,
      title: `Portrait ${String(index)}`,
    }));
    const rows = searchPaletteRows(entries, 'portrait', []);

    expect(rows.filter((row) => row.kind === 'entry')).toHaveLength(12);
  });
});

describe('buildStageEntries', () => {
  it('turns options into keep-open rows that apply then pop', () => {
    const apply = vi.fn();
    const onApplied = vi.fn();
    const entries = buildStageEntries(
      { options: [{ apply, id: 'a', isCurrent: true, label: 'Alpha' }], title: 'Theme' },
      onApplied
    );

    expect(entries[0]).toMatchObject({
      group: 'Theme',
      isCurrent: true,
      keepOpen: true,
      subtitle: 'Current',
      title: 'Alpha',
    });

    entries[0]?.run();
    expect(apply).toHaveBeenCalled();
    expect(onApplied).toHaveBeenCalled();
  });
});

describe('provider row assembly', () => {
  const section = (id: string, count: number, isFetching = false) => ({
    entries: Array.from({ length: count }, (_, index) =>
      makeEntry({ group: id, id: `${id}:${index}`, title: `${id} ${index}` })
    ),
    isError: false,
    isFetching,
    isWaitingForDebounce: false,
    provider: { label: id, providerKey: id },
    retry: vi.fn(),
  });

  it('caps sections in root mode and drops empty settled sections', () => {
    const rows = buildProviderSectionRows([section('workflows', 5), section('boards', 0)]);

    expect(rows.map((row) => row.id)).toEqual([
      'label:provider:workflows',
      getPaletteContributionKey('provider-row', 'workflows:workflows:0'),
      getPaletteContributionKey('provider-row', 'workflows:workflows:1'),
      getPaletteContributionKey('provider-row', 'workflows:workflows:2'),
    ]);
  });

  it('builds one trailing scope row per provider under a "Search in" header', () => {
    const rows = buildScopeRows([{ label: 'Boards', providerKey: 'boards' }], 'sunset');

    expect(rows[0]).toMatchObject({ kind: 'label', label: SEARCH_SCOPE_GROUP });
    expect(rows[1]).toMatchObject({ kind: 'scope', label: 'Search boards for “sunset”', providerKey: 'boards' });
  });
});

describe('searchPaletteRows', () => {
  const entries: PaletteEntry[] = [
    makeEntry({ group: 'Generation', id: 'app.invoke', title: 'Invoke' }),
    makeEntry({ group: 'Generation', id: 'app.invokeFront', title: 'Invoke (Front of Queue)' }),
    makeEntry({ group: 'Navigation', id: 'app.selectCanvasTab', showInEmptyState: true, title: 'Go to Canvas' }),
    makeEntry({ group: 'Settings', id: 'setting.reduceMotion', subtitle: 'Off', title: 'Reduce Motion' }),
  ];

  it('renders the launcher state for an empty query: recents, then navigation only', () => {
    const rows = searchPaletteRows(entries, '', ['setting.reduceMotion']);

    expect(rows.map((row) => row.id)).toEqual([
      'label:Recent',
      'recent:setting.reduceMotion',
      'label:Navigation',
      'app.selectCanvasTab',
    ]);
  });

  it('filters stale recent ids against the live entry list', () => {
    const rows = searchPaletteRows(entries, '', ['no.longer.exists']);

    expect(rows.map((row) => row.id)).toEqual(['label:Navigation', 'app.selectCanvasTab']);
  });

  it('ranks fuzzy matches within their section and drops non-matches', () => {
    const rows = searchPaletteRows(entries, 'invoke', []);

    expect(rows.map((row) => row.id)).toEqual(['label:Generation', 'app.invoke', 'app.invokeFront']);
  });

  it('excludes settings entries in commands-only mode (the ">" alias)', () => {
    const rows = searchPaletteRows(entries, 'o', [], { commandsOnly: true });

    expect(rows.map((row) => row.id)).not.toContain('setting.reduceMotion');
    expect(rows.map((row) => row.id)).toContain('app.invoke');
  });
});

describe('stable active row identity', () => {
  it('preserves the selected action when async rows are inserted before it', () => {
    const selectedRun = vi.fn();
    const selected = makeEntry({ group: 'Commands', id: 'selected', run: selectedRun, title: 'Selected' });
    const initial = searchPaletteRows([selected], 'selected', []);
    const withAsyncInsertion = [
      { id: 'label:provider', kind: 'label' as const, label: 'Provider' },
      { entry: makeEntry({ group: 'Provider', id: 'async', title: 'Async' }), id: 'async-row', kind: 'entry' as const },
      ...initial,
    ];
    const active = resolveActivePaletteRow(getNavigablePaletteRows(withAsyncInsertion), 'selected');

    expect(active?.row.id).toBe('selected');
    if (active?.row.kind === 'entry') {
      active.row.entry.run();
    }
    expect(selectedRun).toHaveBeenCalledOnce();
  });
});
