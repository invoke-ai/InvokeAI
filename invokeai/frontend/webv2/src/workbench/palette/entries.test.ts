import { DEFAULT_PREFERENCES } from '@workbench/settings/store';
import { describe, expect, it, vi } from 'vitest';

import type { PaletteEntry } from './entries';

import {
  buildCatalogCommandEntries,
  buildProviderSectionRows,
  buildScopeRows,
  buildSettingsEntries,
  buildStageEntries,
  searchPaletteRows,
} from './entries';

/**
 * The palette's source aggregation and ranking contract: unimplemented and
 * hotkey-only commands never surface, widget commands follow layout presence,
 * settings rows read/write preferences, and the row list keeps its fixed
 * section order with fuzzy ranking inside each section.
 */

const noop = () => undefined;

const makeEntry = (overrides: Partial<PaletteEntry> & Pick<PaletteEntry, 'id' | 'title' | 'group'>): PaletteEntry => ({
  run: noop,
  ...overrides,
});

describe('buildCatalogCommandEntries', () => {
  const build = (presentWidgetTypeIds: ReadonlySet<string>, execute: (commandId: string) => unknown = noop) =>
    buildCatalogCommandEntries({ customHotkeys: {}, execute, presentWidgetTypeIds });

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
      customHotkeys: { 'app.clearQueue': ['mod+shift+q'] },
      execute,
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
      { openSettingsSection: vi.fn(), patchPreferences }
    );
    const reduceMotion = entries.find((entry) => entry.id === 'setting.reduceMotion');

    expect(reduceMotion).toMatchObject({ keepOpen: true, subtitle: 'Off' });

    reduceMotion?.run();
    expect(patchPreferences).toHaveBeenCalledWith({ reduceMotion: true });
  });

  it('deep-links enum preferences and sections to the settings dialog', () => {
    const openSettingsSection = vi.fn();
    const entries = buildSettingsEntries(DEFAULT_PREFERENCES, { openSettingsSection, patchPreferences: vi.fn() });
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
      { openSettingsSection: vi.fn(), patchPreferences }
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
    provider: { id, label: id },
  });

  it('caps sections in root mode and drops empty settled sections', () => {
    const rows = buildProviderSectionRows([section('workflows', 5), section('boards', 0)]);

    expect(rows.map((row) => row.id)).toEqual([
      'label:provider:workflows',
      'workflows:0',
      'workflows:1',
      'workflows:2',
    ]);
  });

  it('renders uncapped in scoped mode and labels in-flight sections', () => {
    const rows = buildProviderSectionRows([section('workflows', 5, true)], null);

    expect(rows).toHaveLength(6);
    expect(rows[0]).toMatchObject({ kind: 'label', label: 'workflows — Searching…' });
  });

  it('builds one trailing scope row per provider', () => {
    const rows = buildScopeRows([{ id: 'boards', label: 'Boards' }], 'sunset');

    expect(rows[0]).toMatchObject({ kind: 'scope', label: 'Search boards for “sunset”', providerId: 'boards' });
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
});
