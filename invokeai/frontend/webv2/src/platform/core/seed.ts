/**
 * Keep seed policy free of top-level side effects so importing graph compilers cannot pull canvas chunks into
 * settings/palette bundles.
 */

export const SEED_MAX = 4_294_967_295;

export const SEED_MODES = ['random', 'fixed', 'increment', 'decrement'] as const;

export type SeedMode = (typeof SEED_MODES)[number];

/** Direction the seed moves between the entries of one submission's sequence. */
export type SeedStep = -1 | 0 | 1;

export const isSeedMode = (value: unknown): value is SeedMode => SEED_MODES.includes(value as SeedMode);

export const getSeedStep = (mode: SeedMode): SeedStep => (mode === 'fixed' ? 0 : mode === 'decrement' ? -1 : 1);

/** Wraps over the inclusive seed range so decrementing past 0 lands on `SEED_MAX`. */
export const wrapSeed = (seed: number): number => {
  const range = SEED_MAX + 1;

  return ((seed % range) + range) % range;
};

export interface SeedSequenceInput {
  batchCount: number;
  /** Concrete prompts the submission carries; one unless the host expands a prompt into several. */
  promptCount: number;
  seedBehaviour: 'per-image' | 'per-iteration';
  seedMode: SeedMode;
}

/** Fixed consumes one seed; otherwise consume per iteration, or per image when prompts do not share seeds. */
export const getSeedSequenceLength = ({
  batchCount,
  promptCount,
  seedBehaviour,
  seedMode,
}: SeedSequenceInput): number =>
  seedMode === 'fixed' ? 1 : promptCount > 1 && seedBehaviour === 'per-image' ? promptCount * batchCount : batchCount;

export interface SeedSubmissionPlan {
  /** The last sequence entry the submission uses. */
  lastSeed: number;
  /** What the editable seed becomes once the submission is queued; null when the mode leaves it alone. */
  nextSeed: number | null;
  seedMode: SeedMode;
  sequenceLength: number;
  startSeed: number;
  step: SeedStep;
}

/** Only stepping modes advance the editable seed; random preserves it and fixed reuses it. */
export const planSeedSubmission = ({
  startSeed,
  ...sequence
}: SeedSequenceInput & { startSeed: number }): SeedSubmissionPlan => {
  const step = getSeedStep(sequence.seedMode);
  const sequenceLength = getSeedSequenceLength(sequence);
  const advances = sequence.seedMode === 'increment' || sequence.seedMode === 'decrement';

  return {
    lastSeed: wrapSeed(startSeed + step * (sequenceLength - 1)),
    nextSeed: advances ? wrapSeed(startSeed + step * sequenceLength) : null,
    seedMode: sequence.seedMode,
    sequenceLength,
    startSeed,
    step,
  };
};
