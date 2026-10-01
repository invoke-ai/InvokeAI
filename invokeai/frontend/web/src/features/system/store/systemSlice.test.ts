import { describe, expect, it } from 'vitest';

import { systemSliceConfig } from './systemSlice';

describe('systemSliceConfig persisted state migration', () => {
  const migrate = systemSliceConfig.persistConfig?.migrate;

  it('adds starred media protection disabled when migrating the current main state', () => {
    expect(migrate).toBeDefined();
    const state: Record<string, unknown> = {
      ...systemSliceConfig.getInitialState(),
      _version: 3,
    };
    delete state.shouldProtectStarredMedia;

    const result = migrate?.(state);

    expect(result?._version).toBe(4);
    expect(result?.shouldProtectStarredMedia).toBe(false);
  });

  it('preserves the branch-only starred image preference under the media setting', () => {
    expect(migrate).toBeDefined();
    const state: Record<string, unknown> = {
      ...systemSliceConfig.getInitialState(),
      _version: 4,
      shouldProtectStarredImages: true,
    };
    delete state.shouldProtectStarredMedia;

    const result = migrate?.(state);

    expect(result?._version).toBe(4);
    expect(result?.shouldProtectStarredMedia).toBe(true);
  });
});
