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
  it('does not auto-select a low-noise expert when it is the only Wan model installed', () => {
    const dispatch = vi.fn();
    handleMainModels([wanLowExpert], makeState(), dispatch, log);

    // Nothing selectable, so the selection is left null rather than pointed at a model
    // that cannot load. (`model` is already null, so no clear is dispatched either.)
    expect(dispatch).not.toHaveBeenCalled();
  });

  it('auto-selects the high-noise expert over the low-noise one regardless of order', () => {
    const dispatch = vi.fn();
    handleMainModels([wanLowExpert, wanHighExpert], makeState(), dispatch, log);

    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(dispatch).toHaveBeenCalledWith(modelSelected(wanHighExpert));
  });
});
