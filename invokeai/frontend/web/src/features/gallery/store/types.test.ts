import type { S } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import { describe, expect, test } from 'vitest';

import { galleryLayoutModeChanged, gallerySliceConfig } from './gallerySlice';
import type { BoardRecordOrderBy } from './types';

describe('Gallery Types', () => {
  // Ensure zod types match OpenAPI types
  test('BoardRecordOrderBy', () => {
    assert<Equals<BoardRecordOrderBy, S['BoardRecordOrderBy']>>();
  });

  test('defaults gallery layout mode to grid', () => {
    expect(gallerySliceConfig.getInitialState().galleryLayoutMode).toBe('grid');
  });

  test('migrates legacy gallery state to grid layout mode', () => {
    const migrate = gallerySliceConfig.persistConfig?.migrate;
    expect(migrate).toBeDefined();

    const legacyState: Partial<ReturnType<typeof gallerySliceConfig.getInitialState>> =
      gallerySliceConfig.getInitialState();
    delete legacyState.galleryLayoutMode;

    expect(migrate?.(legacyState)?.galleryLayoutMode).toBe('grid');
  });

  test('sets gallery layout mode', () => {
    const result = gallerySliceConfig.slice.reducer(undefined, galleryLayoutModeChanged('masonry'));
    expect(result.galleryLayoutMode).toBe('masonry');
  });
});
