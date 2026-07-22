import { DEFAULT_PREFERENCES } from '@workbench/settings/store';
import { describe, expect, it, vi } from 'vitest';

import type { PaletteEntry } from './entries';

import { buildCatalogCommandEntries, buildSettingsEntries, searchPaletteRows } from './entries';

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
});
