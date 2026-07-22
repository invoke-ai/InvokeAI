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
  SEARCH_SCOPE_GROUP,
  searchPaletteRows,
  resolveActivePaletteRow,
} from './entries';

/**
 * The palette's source aggregation and ranking contract: unimplemented and
 * hotkey-only commands never surface, widget commands follow layout presence,
 * settings rows read/write preferences, and the row list keeps its fixed
 * section order with fuzzy ranking inside each section.
 */

const noop = () => undefined;
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
    });

  it('surfaces implemented app commands with palette-facing titles and groups', () => {
    const entries = build(new Set());
    const invoke = entries.find((entry) => entry.id === 'app.invoke');
    const generateTab = entries.find((entry) => entry.id === 'app.selectGenerateTab');

    expect(invoke).toMatchObject({ group: 'Generation', title: 'Invoke' });
    expect(generateTab).toMatchObject({ group: 'Navigation', showInEmptyState: true, title: 'Go to Generate' });
  });

  it('hides unimplemented and hotkey-only commands', () => {
    const ids = build(new Set(['canvas', 'gallery', 'preview', 'workflow'])).map((entry) => entry.id);

    expect(ids).not.toContain('app.selectUpscalingTab');
    expect(ids).not.toContain('gallery.galleryNavLeft');
    expect(ids).not.toContain('app.promptWeightUp');
    expect(ids).not.toContain('app.openCommandPalette');
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
      formatHotkey: formatHotkeyForPlatform,
      presentWidgetTypeIds: new Set(),
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
      { ...settingsEntryDeps, openSettingsSection: vi.fn(), patchPreferences }
    );
    const reduceMotion = entries.find((entry) => entry.id === 'setting.reduceMotion');

    expect(reduceMotion).toMatchObject({ keepOpen: true, subtitle: 'Off' });

    reduceMotion?.run();
    expect(patchPreferences).toHaveBeenCalledWith({ reduceMotion: true });
  });

  it('deep-links enum preferences and sections to the settings dialog', () => {
    const openSettingsSection = vi.fn();
    const entries = buildSettingsEntries(DEFAULT_PREFERENCES, {
      ...settingsEntryDeps,
      openSettingsSection,
      patchPreferences: vi.fn(),
    });
    const theme = entries.find((entry) => entry.id === 'setting.themeId');
    const workflowSection = entries.find((entry) => entry.id === 'settings.section.workflow');

    expect(theme?.subtitle).toBe('Classic');

    theme?.run();
    expect(openSettingsSection).toHaveBeenCalledWith('appearance');

    workflowSection?.run();
    expect(openSettingsSection).toHaveBeenCalledWith('workflow');
  });

  it('carries a value-picker stage on enum preferences that applies and marks the current value', () => {
    const patchPreferences = vi.fn();
    const entries = buildSettingsEntries(
      { ...DEFAULT_PREFERENCES, workflowEdgeStyle: 'curved' },
      { ...settingsEntryDeps, openSettingsSection: vi.fn(), patchPreferences }
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

  it('wires theme stage preview hooks only when the host provides them', () => {
    const previewTheme = vi.fn();
    const clearThemePreview = vi.fn();
    const withPreview = buildSettingsEntries(DEFAULT_PREFERENCES, {
      ...settingsEntryDeps,
      clearThemePreview,
      openSettingsSection: vi.fn(),
      patchPreferences: vi.fn(),
      previewTheme,
    });
    const theme = withPreview.find((entry) => entry.id === 'setting.themeId');

    theme?.stage?.preview?.('light');
    theme?.stage?.clearPreview?.();
    expect(previewTheme).toHaveBeenCalledWith('light');
    expect(clearThemePreview).toHaveBeenCalled();

    const withoutPreview = buildSettingsEntries(DEFAULT_PREFERENCES, {
      ...settingsEntryDeps,
      openSettingsSection: vi.fn(),
      patchPreferences: vi.fn(),
    });
    const bareTheme = withoutPreview.find((entry) => entry.id === 'setting.themeId');

    expect(bareTheme?.stage?.preview).toBeUndefined();
    expect(bareTheme?.stage?.clearPreview).toBeUndefined();
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

    expect(entries[0]).toMatchObject({ group: 'Theme', keepOpen: true, subtitle: 'Current', title: 'Alpha' });

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
    isFetching,
    provider: { label: id, providerKey: id },
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

  it('renders uncapped in scoped mode and labels in-flight sections', () => {
    const rows = buildProviderSectionRows([section('workflows', 5, true)], null);

    expect(rows).toHaveLength(6);
    expect(rows[0]).toMatchObject({ kind: 'label', label: 'workflows — Searching…' });
  });

  it('builds one trailing scope row per provider', () => {
    const rows = buildScopeRows([{ label: 'Boards', providerKey: 'boards' }], 'sunset');

    expect(rows[0]).toMatchObject({ kind: 'scope', label: 'Search boards for “sunset”', providerKey: 'boards' });
  });

  it('labels scope rows as by-date searches when only a date filter is active', () => {
    const rows = buildScopeRows([{ label: 'Images', providerKey: 'images' }], '');

    expect(rows[0]).toMatchObject({ kind: 'scope', label: 'Search images by date', providerKey: 'images' });
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

  it('keeps the fixed section order regardless of match strength', () => {
    const rows = searchPaletteRows(
      [
        makeEntry({ group: 'Settings', id: 'setting.a', title: 'Motion' }),
        makeEntry({ group: 'Generation', id: 'command.a', title: 'Prompt Motion Nudge' }),
      ],
      'motion',
      []
    );
    const labels = rows.filter((row) => row.kind === 'label').map((row) => row.id);

    expect(labels).toEqual(['label:Generation', 'label:Settings']);
  });

  it('returns no rows for a junk query', () => {
    expect(searchPaletteRows(entries, 'zzzzqqq', [])).toEqual([]);
  });

  it('excludes settings entries in commands-only mode (the ">" alias)', () => {
    const rows = searchPaletteRows(entries, 'o', [], { commandsOnly: true });

    expect(rows.map((row) => row.id)).not.toContain('setting.reduceMotion');
    expect(rows.map((row) => row.id)).toContain('app.invoke');
  });

  it('lists every non-setting command for a bare commands scope', () => {
    const rows = searchPaletteRows(entries, '', [], { commandsOnly: true, showAllOnEmpty: true });

    expect(rows.filter((row) => row.kind === 'entry').map((row) => row.entry.id)).toEqual([
      'app.invoke',
      'app.invokeFront',
      'app.selectCanvasTab',
    ]);
  });

  it('renders the search-scope section last in the empty launcher', () => {
    const rows = searchPaletteRows(
      [
        ...entries,
        makeEntry({ group: SEARCH_SCOPE_GROUP, id: 'scope.images', showInEmptyState: true, title: 'Search images…' }),
      ],
      '',
      []
    );

    expect(rows.filter((row) => row.kind === 'label').map((row) => row.id)).toEqual([
      'label:Navigation',
      `label:${SEARCH_SCOPE_GROUP}`,
    ]);
    expect(rows.at(-1)?.id).toBe('scope.images');
  });

  it('includes search-scope commands in commands-only mode', () => {
    const rows = searchPaletteRows(
      [...entries, makeEntry({ group: SEARCH_SCOPE_GROUP, id: 'scope.images', title: 'Search images…' })],
      '',
      [],
      { commandsOnly: true, showAllOnEmpty: true }
    );

    expect(rows.filter((row) => row.kind === 'entry').map((row) => row.entry.id)).toContain('scope.images');
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
    const active = resolveActivePaletteRow(withAsyncInsertion, 'selected');

    expect(active?.row.id).toBe('selected');
    if (active?.row.kind === 'entry') {
      active.row.entry.run();
    }
    expect(selectedRun).toHaveBeenCalledOnce();
  });
});
