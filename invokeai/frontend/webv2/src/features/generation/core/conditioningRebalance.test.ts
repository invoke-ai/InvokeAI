import { describe, expect, it } from 'vitest';

import {
  adjustRebalanceWeight,
  barFillFraction,
  BUILTIN_REBALANCE_PRESETS,
  DEFAULT_KREA2_REBALANCE_MULTIPLIER,
  DEFAULT_KREA2_REBALANCE_WEIGHTS,
  getRebalanceBarScale,
  getRebalanceSparklinePath,
  isValidKrea2RebalanceWeights,
  KREA2_REBALANCE_WEIGHT_COUNT,
  KREA2_TAP_LAYERS,
  matchRebalancePreset,
  NEUTRAL_KREA2_REBALANCE_WEIGHTS,
  normalizeRebalancePresets,
  parseRebalanceWeights,
  REBALANCE_WEIGHT_STEP,
  REBALANCE_WEIGHT_TRACK_MAX,
  serializeRebalanceWeights,
  weightFromTrackFraction,
} from './conditioningRebalance';

describe('KREA2_TAP_LAYERS', () => {
  it('matches the encoder layers the backend taps', () => {
    // KREA2_SELECT_LAYERS in invokeai/backend/krea2/sampling_utils.py.
    expect([...KREA2_TAP_LAYERS]).toEqual([2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]);
  });

  it('has one entry per weight', () => {
    expect(KREA2_TAP_LAYERS).toHaveLength(KREA2_REBALANCE_WEIGHT_COUNT);
  });
});

describe('parseRebalanceWeights', () => {
  it('parses the backend default vector', () => {
    expect(parseRebalanceWeights(DEFAULT_KREA2_REBALANCE_WEIGHTS)).toEqual([1, 1, 1, 1, 1, 1, 1, 2.5, 5, 1.1, 4, 1]);
  });

  it('accepts the decimal forms Python float() accepts', () => {
    expect(parseRebalanceWeights('1,.5,+2.5,1e-3,2.,0,1,1,1,1,1,1')).toEqual([
      1, 0.5, 2.5, 0.001, 2, 0, 1, 1, 1, 1, 1, 1,
    ]);
  });

  it('tolerates surrounding whitespace', () => {
    expect(parseRebalanceWeights(' 1.0 , 1.0 ,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 ')).toHaveLength(12);
  });

  it('rejects hex, which Number() accepts but Python float() does not', () => {
    // The string is forwarded verbatim to the node, so accepting it here would only
    // move the failure to mid-generation.
    expect(parseRebalanceWeights('0x10,1,1,1,1,1,1,1,1,1,1,1')).toBeNull();
    expect(parseRebalanceWeights('0b11,1,1,1,1,1,1,1,1,1,1,1')).toBeNull();
    expect(parseRebalanceWeights('0o17,1,1,1,1,1,1,1,1,1,1,1')).toBeNull();
  });

  it('rejects non-finite values', () => {
    expect(parseRebalanceWeights('1e999,1,1,1,1,1,1,1,1,1,1,1')).toBeNull();
    expect(parseRebalanceWeights('NaN,1,1,1,1,1,1,1,1,1,1,1')).toBeNull();
    expect(parseRebalanceWeights('Infinity,1,1,1,1,1,1,1,1,1,1,1')).toBeNull();
  });

  it('rejects empty entries and the wrong count', () => {
    expect(parseRebalanceWeights('1,,1,1,1,1,1,1,1,1,1,1')).toBeNull();
    expect(parseRebalanceWeights('1,1,1')).toBeNull();
    expect(parseRebalanceWeights(`${DEFAULT_KREA2_REBALANCE_WEIGHTS},1.0`)).toBeNull();
    expect(parseRebalanceWeights('')).toBeNull();
  });

  it('backs isValidKrea2RebalanceWeights', () => {
    expect(isValidKrea2RebalanceWeights(DEFAULT_KREA2_REBALANCE_WEIGHTS)).toBe(true);
    expect(isValidKrea2RebalanceWeights('0x10,1,1,1,1,1,1,1,1,1,1,1')).toBe(false);
  });
});

describe('serializeRebalanceWeights', () => {
  it('round-trips the backend default string exactly', () => {
    const parsed = parseRebalanceWeights(DEFAULT_KREA2_REBALANCE_WEIGHTS);

    expect(parsed).not.toBeNull();
    expect(serializeRebalanceWeights(parsed ?? [])).toBe(DEFAULT_KREA2_REBALANCE_WEIGHTS);
  });

  it('keeps drag output free of float dust', () => {
    expect(serializeRebalanceWeights([0.1 + 0.2])).toBe('0.3');
    expect(serializeRebalanceWeights([2])).toBe('2.0');
  });

  it('produces a neutral vector that parses back to all ones', () => {
    expect(parseRebalanceWeights(NEUTRAL_KREA2_REBALANCE_WEIGHTS)).toEqual(Array.from({ length: 12 }, () => 1));
  });
});

describe('bar geometry', () => {
  it('uses the nominal ceiling until a weight overshoots it', () => {
    expect(getRebalanceBarScale([1, 2, 5])).toBe(REBALANCE_WEIGHT_TRACK_MAX);
    expect(getRebalanceBarScale([1, 12.5, 5])).toBe(12.5);
  });

  it('maps the track top to full scale and the bottom to zero', () => {
    expect(weightFromTrackFraction(0, 8)).toBe(8);
    expect(weightFromTrackFraction(1, 8)).toBe(0);
    expect(weightFromTrackFraction(0.5, 8)).toBe(4);
  });

  it('clamps pointer positions that leave the track', () => {
    expect(weightFromTrackFraction(-0.5, 8)).toBe(8);
    expect(weightFromTrackFraction(1.5, 8)).toBe(0);
  });

  it('inverts barFillFraction on the step grid', () => {
    const scale = REBALANCE_WEIGHT_TRACK_MAX;

    for (let step = 0; step <= scale / REBALANCE_WEIGHT_STEP; step += 1) {
      const weight = Math.round(step * REBALANCE_WEIGHT_STEP * 10) / 10;

      expect(weightFromTrackFraction(1 - barFillFraction(weight, scale), scale)).toBe(weight);
    }
  });

  it('snaps adjustments back onto the step grid', () => {
    expect(adjustRebalanceWeight(1.1, REBALANCE_WEIGHT_STEP, 8)).toBe(1.2);
    expect(adjustRebalanceWeight(2.5, -REBALANCE_WEIGHT_STEP * 2, 8)).toBe(2.3);
    expect(adjustRebalanceWeight(0, -1, 8)).toBe(0);
    expect(adjustRebalanceWeight(8, 1, 8)).toBe(8);
  });
});

describe('getRebalanceSparklinePath', () => {
  it('emits one point per weight, taller weights higher up', () => {
    const path = getRebalanceSparklinePath([0, 8], 100, 14);
    const [start, end] = path.split(' L');

    expect(start).toBe('M0 13');
    expect(end).toBe('100 1');
  });

  it('returns an empty path for an empty vector', () => {
    expect(getRebalanceSparklinePath([], 100, 14)).toBe('');
  });
});

describe('presets', () => {
  it('ships the backend defaults and a neutral pass', () => {
    expect(BUILTIN_REBALANCE_PRESETS.map((preset) => preset.id)).toEqual(['default', 'neutral']);
    expect(BUILTIN_REBALANCE_PRESETS[0]?.weights).toBe(DEFAULT_KREA2_REBALANCE_WEIGHTS);
    expect(BUILTIN_REBALANCE_PRESETS[0]?.multiplier).toBe(DEFAULT_KREA2_REBALANCE_MULTIPLIER);
    expect(BUILTIN_REBALANCE_PRESETS[1]?.multiplier).toBe(1);
  });

  it('matches on the parsed vector, not the string spelling', () => {
    expect(matchRebalancePreset(BUILTIN_REBALANCE_PRESETS, DEFAULT_KREA2_REBALANCE_WEIGHTS, 4)).toBe('default');
    expect(matchRebalancePreset(BUILTIN_REBALANCE_PRESETS, '1,1,1,1,1,1,1,2.5,5,1.1,4,1', 4)).toBe('default');
  });

  it('reports no match once the gain or a weight is edited', () => {
    expect(matchRebalancePreset(BUILTIN_REBALANCE_PRESETS, DEFAULT_KREA2_REBALANCE_WEIGHTS, 4.5)).toBeNull();
    expect(
      matchRebalancePreset(BUILTIN_REBALANCE_PRESETS, '1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.5', 4)
    ).toBeNull();
    expect(matchRebalancePreset(BUILTIN_REBALANCE_PRESETS, 'not weights', 4)).toBeNull();
  });

  it('drops persisted entries that no longer parse', () => {
    const presets = normalizeRebalancePresets([
      { id: 'a', label: 'Keep', multiplier: 2, weights: NEUTRAL_KREA2_REBALANCE_WEIGHTS },
      { id: 'b', label: 'Bad weights', multiplier: 2, weights: '1,2,3' },
      { id: 'c', label: '   ', multiplier: 2, weights: NEUTRAL_KREA2_REBALANCE_WEIGHTS },
      { id: 'd', label: 'Bad gain', multiplier: Number.NaN, weights: NEUTRAL_KREA2_REBALANCE_WEIGHTS },
      null,
      'nope',
    ]);

    expect(presets.map((preset) => preset.id)).toEqual(['a']);
  });

  it('drops duplicate ids and ids that shadow a built-in', () => {
    const presets = normalizeRebalancePresets([
      { id: 'a', label: 'First', multiplier: 1, weights: NEUTRAL_KREA2_REBALANCE_WEIGHTS },
      { id: 'a', label: 'Duplicate', multiplier: 1, weights: NEUTRAL_KREA2_REBALANCE_WEIGHTS },
      { id: 'default', label: 'Shadow', multiplier: 1, weights: NEUTRAL_KREA2_REBALANCE_WEIGHTS },
    ]);

    expect(presets).toEqual([{ id: 'a', label: 'First', multiplier: 1, weights: NEUTRAL_KREA2_REBALANCE_WEIGHTS }]);
  });

  it('returns an empty list for a non-array blob', () => {
    expect(normalizeRebalancePresets(undefined)).toEqual([]);
    expect(normalizeRebalancePresets({ id: 'a' })).toEqual([]);
  });
});
