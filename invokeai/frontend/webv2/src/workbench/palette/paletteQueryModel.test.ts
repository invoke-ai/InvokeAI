import { describe, expect, it, vi } from 'vitest';

import type { PaletteSearchProvider } from './entries';

import { derivePaletteQueryModel } from './paletteQueryModel';
import { createInitialPaletteState, enterPaletteScope, type PaletteState } from './paletteState';

const provider = (providerKey: string, supportsCreatedAtRange = false): PaletteSearchProvider => ({
  contextKey: providerKey,
  label: providerKey,
  providerKey,
  search: vi.fn(() => []),
  supportsCreatedAtRange,
});

const queriedState = (query: string, debouncedQuery = query): PaletteState => ({
  ...createInitialPaletteState(),
  debouncedQuery,
  query,
});

describe('derivePaletteQueryModel', () => {
  it('falls back to root mode when a scoped provider disappears', () => {
    const result = derivePaletteQueryModel({ providers: [], state: enterPaletteScope('missing', 'query') });

    expect(result.resolvedMode).toEqual({ kind: 'root' });
    expect(result.scopeProvider).toBeNull();
    expect(result.scopeProviderKey).toBeNull();
  });

  it('disables date parsing and providers in commands mode', () => {
    const result = derivePaletteQueryModel({
      providers: [provider('images', true)],
      state: queriedState('>from:2026-07-14'),
    });

    expect(result.isCommandsScope).toBe(true);
    expect(result.localQuery).toBe('from:2026-07-14');
    expect(result.dateParse).toBeNull();
    expect(result.shouldSearchProviders).toBe(false);
    expect(result.activeProviders).toEqual([]);
  });

  it('enforces the root minimum query length but searches every scoped query', () => {
    const entities = provider('entities');

    expect(derivePaletteQueryModel({ providers: [entities], state: queriedState('a') }).shouldSearchProviders).toBe(
      false
    );
    expect(derivePaletteQueryModel({ providers: [entities], state: queriedState('ab') }).shouldSearchProviders).toBe(
      true
    );
    expect(
      derivePaletteQueryModel({ providers: [entities], state: enterPaletteScope('entities', '') }).shouldSearchProviders
    ).toBe(true);
    expect(
      derivePaletteQueryModel({ providers: [entities], state: enterPaletteScope('entities', 'a') })
        .shouldSearchProviders
    ).toBe(true);
  });

  it('runs only range-capable providers for a pure date query', () => {
    const images = provider('images', true);
    const workflows = provider('workflows');
    const result = derivePaletteQueryModel({
      providers: [images, workflows],
      state: queriedState('from:2026-07-14'),
    });

    expect(result.isPureDateQuery).toBe(true);
    expect(result.shouldSearchProviders).toBe(true);
    expect(result.activeProviders).toEqual([images]);
    expect(result.liveProviderQuery).toEqual({ range: { from: '2026-07-14', to: undefined }, text: '' });
  });

  it('reports when the live query is ahead of the debounced provider query', () => {
    const result = derivePaletteQueryModel({ providers: [provider('entities')], state: queriedState('new', 'old') });

    expect(result.isWaitingForDebounce).toBe(true);
    expect(result.liveProviderQuery.text).toBe('new');
    expect(result.providerQuery.text).toBe('old');
  });
});
