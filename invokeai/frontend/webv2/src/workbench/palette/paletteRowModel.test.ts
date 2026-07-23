import type { TFunction } from 'i18next';

import { describe, expect, it, vi } from 'vitest';

import type { PaletteEntry, PaletteSearchProvider, ProviderResultSection } from './entries';

import { derivePaletteQueryModel } from './paletteQueryModel';
import { buildCommandPaletteRows, buildStageRows } from './paletteRowModel';
import { createInitialPaletteState, enterPaletteStage, type PaletteState } from './paletteState';

const t = ((key: string, options?: Record<string, unknown>) =>
  typeof options?.label === 'string' ? `${key}:${options.label}` : key) as TFunction;
const entry = (id: string, showInEmptyState = true): PaletteEntry => ({
  group: 'Commands',
  id,
  isPersistentRecent: true,
  run: vi.fn(),
  showInEmptyState,
  title: id,
});
const provider = (supportsCreatedAtRange = false): PaletteSearchProvider => ({
  contextKey: 'images',
  label: 'Images',
  providerKey: 'images',
  search: vi.fn(() => []),
  supportsCreatedAtRange,
});
const queriedState = (query: string): PaletteState => ({
  ...createInitialPaletteState(),
  debouncedQuery: query,
  query,
});
const build = ({
  entries = [],
  providers = [],
  providerSections = [],
  state = createInitialPaletteState(),
}: {
  entries?: PaletteEntry[];
  providers?: PaletteSearchProvider[];
  providerSections?: ProviderResultSection[];
  state?: PaletteState;
}) => {
  const queryModel = derivePaletteQueryModel({ providers, state });

  return buildCommandPaletteRows({
    enterScope: vi.fn(),
    entries,
    onCompleteDateSuggestion: vi.fn(),
    onStageApplied: vi.fn(),
    providers,
    providerSections,
    queryModel,
    recentIds: entries.map((candidate) => candidate.id),
    t,
  });
};

describe('buildCommandPaletteRows', () => {
  it('suppresses recents for pure-date queries and renders only date-capable provider paths', () => {
    const images = provider(true);
    const providerEntry = entry('dated-image', false);
    const providerSections: ProviderResultSection[] = [
      {
        entries: [providerEntry],
        isError: false,
        isFetching: false,
        isWaitingForDebounce: false,
        provider: images,
        retry: vi.fn(),
      },
    ];

    const rows = build({
      entries: [entry('recent-command')],
      providers: [images, { ...provider(), label: 'Workflows', providerKey: 'workflows' }],
      providerSections,
      state: queriedState('from:2026-07-14'),
    });

    expect(rows.some((row) => row.id.includes('recent-command'))).toBe(false);
    expect(rows.some((row) => row.kind === 'entry' && row.entry.id === providerEntry.id)).toBe(true);
    expect(rows.filter((row) => row.kind === 'scope').map((row) => row.providerKey)).toEqual(['images']);
  });

  it('renders date suggestions before other root results', () => {
    const rows = build({ providers: [provider(true)], state: queriedState('from:') });

    expect(rows[0]).toMatchObject({ id: 'label:date-suggestions', kind: 'label' });
    expect(rows.some((row) => row.id === 'date-suggestion:from:today')).toBe(true);
  });

  it('keeps stage rows isolated from root entries and providers', () => {
    const stage = {
      options: [{ apply: vi.fn(), id: 'dark', isCurrent: true, label: 'Dark' }],
      title: 'Theme',
    };
    const rows = build({
      entries: [entry('root-command')],
      providers: [provider()],
      state: enterPaletteStage(stage, null),
    });

    expect(rows.some((row) => row.id.includes('root-command'))).toBe(false);
    expect(rows.some((row) => row.id === 'stage:dark')).toBe(true);
  });

  it('adds the host-appropriate syntax hint to the empty launcher', () => {
    const commandsOnly = build({ entries: [entry('command')] });
    const withDates = build({ entries: [entry('command')], providers: [provider(true)] });

    expect(commandsOnly.at(-1)).toMatchObject({ label: 'commandPalette.syntax.commands' });
    expect(withDates.at(-1)).toMatchObject({ label: 'commandPalette.syntax.commandsAndDates' });
  });
});

describe('buildStageRows', () => {
  const stage = {
    options: [
      { apply: vi.fn(), id: 'dark', isCurrent: true, label: 'Dark' },
      { apply: vi.fn(), id: 'light', isCurrent: false, label: 'Light' },
    ],
    title: 'Theme',
  };

  it('shows every option for an empty query', () => {
    const rows = buildStageRows(stage, vi.fn(), '');

    expect(rows.filter((row) => row.kind === 'entry').map((row) => row.id)).toEqual(['stage:dark', 'stage:light']);
  });

  it('filters options for a non-empty query', () => {
    const rows = buildStageRows(stage, vi.fn(), 'li');

    expect(rows.filter((row) => row.kind === 'entry').map((row) => row.id)).toEqual(['stage:light']);
  });

  it('marks the current option with the localized subtitle when a translator is supplied', () => {
    const rows = buildStageRows(stage, vi.fn(), '', t);
    const current = rows.find((row) => row.id === 'stage:dark');

    expect(current?.kind === 'entry' && current.entry.subtitle).toBe('commandPalette.settings.current');
  });
});
