import type { RootState } from 'app/store/store';
import { modelSelected } from 'features/parameters/store/actions';
import type { AnyModelConfig } from 'services/api/types';
import { describe, expect, it, vi } from 'vitest';

import { handleMainModels } from './modelsLoaded';

/**
 * A Wan low-noise expert belongs in the Transformer (Low Noise) slot; running it alone as
 * the primary main is accepted by the loader since #9505 but gives visibly worse output.
 * So no path that chooses a primary main *on the user's behalf* should reach for one
 * while its partner is installed. This listener is the least visible of the three: it
 * fires on every `getModelConfigs` fulfilment and swaps the selection silently.
 *
 * Which is exactly why it must not treat "hidden" as "uninstalled" — see the last test.
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

const sdxlModel = {
  key: 'sdxl',
  hash: 'h',
  name: 'SDXL',
  base: 'sdxl',
  type: 'main',
  format: 'checkpoint',
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

  it('does not swap the user off a selected low expert when its partner is installed', () => {
    // The regression this guards: hiding is a *visibility* rule, and the availability
    // check must not use it. Otherwise installing the high-noise partner reads as "your
    // model was uninstalled" and silently moves the user to another model — firing the
    // base-changed cascade for what was just a file install.
    const dispatch = vi.fn();
    const state = { params: { model: wanLowExpert } } as unknown as RootState;
    handleMainModels([wanLowExpert, wanHighExpert, sdxlModel], state, dispatch, log);

    expect(dispatch).not.toHaveBeenCalled();
  });

  it('never auto-selects a hidden low expert when a partner exists', () => {
    const dispatch = vi.fn();
    handleMainModels([wanLowExpert, wanHighExpert, sdxlModel], makeState(), dispatch, log);

    expect(dispatch).toHaveBeenCalledWith(modelSelected(sdxlModel));
  });
});
