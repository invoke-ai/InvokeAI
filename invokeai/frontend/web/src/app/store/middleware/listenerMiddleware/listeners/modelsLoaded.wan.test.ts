import type { RootState } from 'app/store/store';
import { modelSelected } from 'features/parameters/store/actions';
import type { AnyModelConfig } from 'services/api/types';
import { describe, expect, it, vi } from 'vitest';

import { handleMainModels } from './modelsLoaded';

/**
 * A Wan low-noise expert is refused by the loader as a primary main ("An unpaired Wan
 * A14B model must be the high-noise expert"), so no path that chooses a primary main on
 * the user's behalf may reach for one. This listener is the least visible of the three:
 * it fires on every `getModelConfigs` fulfilment and swaps the selection silently.
 */

const wanHighExpert = {
  key: 'wan-high',
  hash: 'h',
  name: 'Wan2.2-T2V-A14B-HIGH',
  base: 'wan',
  type: 'main',
  format: 'checkpoint',
  variant: 't2v_a14b',
  expert: 'high',
} as unknown as AnyModelConfig;

const wanLowExpert = {
  ...wanHighExpert,
  key: 'wan-low',
  name: 'Wan2.2-T2V-A14B-LOW',
  expert: 'low',
} as unknown as AnyModelConfig;

const makeState = () => ({ params: { model: null } }) as unknown as RootState;

const log = { debug: vi.fn(), info: vi.fn(), error: vi.fn(), warn: vi.fn() } as never;

describe('handleMainModels — Wan low-noise experts', () => {
  it('auto-selects the high-noise expert over the low-noise one regardless of order', () => {
    const dispatch = vi.fn();
    handleMainModels([wanLowExpert, wanHighExpert], makeState(), dispatch, log);

    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(dispatch).toHaveBeenCalledWith(modelSelected(wanHighExpert));
  });

  it('does offer the low-noise expert when its partner is not installed', () => {
    // Since #9505 the loader runs an unpaired low expert with a warning instead of
    // refusing it, so hiding it here with no alternative would be a dead end — the
    // user's only Wan model would be missing from every picker.
    const dispatch = vi.fn();
    handleMainModels([wanLowExpert], makeState(), dispatch, log);

    expect(dispatch).toHaveBeenCalledWith(modelSelected(wanLowExpert));
  });

  it('treats a different-variant high expert as no partner at all', () => {
    // An I2V high expert cannot pair with a T2V low one — the loader rejects the
    // variant mismatch — so it must not be the reason the T2V low expert is hidden.
    const dispatch = vi.fn();
    const i2vHigh = { ...wanHighExpert, key: 'wan-i2v-high', variant: 'i2v_a14b' } as unknown as AnyModelConfig;
    handleMainModels([wanLowExpert, i2vHigh], makeState(), dispatch, log);

    // Both remain offerable; the sort leaves the list order, so the low expert is first.
    expect(dispatch).toHaveBeenCalledWith(modelSelected(wanLowExpert));
  });
});
