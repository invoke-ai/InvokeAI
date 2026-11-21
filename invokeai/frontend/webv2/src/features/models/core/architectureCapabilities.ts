/**
 * The backend's architecture table (`GET /api/v2/models/capabilities`) is authoritative. The handwritten DTO is pinned
 * to it by `__fixtures__/architectureCapabilities.json`; `data/` pushes fetched tables into this React-free registry.
 */

import { createListenerChannel } from '@platform/state/externalStoreCore';

/** A capabilities row exactly as the backend serialises it. snake_case, unions kept open where the backend's are. */
export interface ArchitectureCapabilitiesRow {
  base: string;
  /** Null for the architecture's own row. A variant row overrides it. */
  variant: string | null;
  features: {
    negative_prompt: { visible: boolean; usage: 'always' | 'cfg-gated' | 'never' };
    dimension_grid: number;
    guidance_label: 'CFG' | 'Guidance' | (string & {});
    /** The floor the denoise node enforces on the guidance field; `0` where it enforces none. */
    guidance_min: number;
    /** The ceiling the denoise node enforces, or `null` where it enforces none. */
    guidance_max: number | null;
    /** A variant row may carry its own. */
    scheduler_set: 'standard' | 'flow' | 'flow-no-lcm' | 'anima' | null;
    scheduler_applies_to_graph: boolean;
    control_kinds: string[];
    max_reference_images: number;
    reference_images_require_variant: string | null;
    reference_images_require_vision_tower: boolean;
    supports_regional_guidance: boolean;
    regional_negative: boolean;
    clip_skip_max: number | null;
    supports_seamless: boolean;
    supports_cfg_rescale: boolean;
    sd_vae_override: boolean;
    color_compensation: boolean;
    vae_precision: boolean;
  };
  defaults: {
    vae: string | null;
    vae_precision: 'fp16' | 'fp32' | null;
    scheduler: string | null;
    steps: number | null;
    cfg_scale: number | null;
    cfg_rescale_multiplier: number | null;
    width: number | null;
    height: number | null;
    guidance: number | null;
    cpu_only: boolean | null;
    fp8_storage: boolean | null;
  } | null;
  /** Null restricts VAE compatibility to the model's own base; channel restrictions are optional. */
  vae: {
    accepted: { base: string; latent_channels: number | null }[];
  } | null;
}

const isObject = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

/** Rejects a row whose blocks readers dereference, so a malformed table is never published. */
const assertRow = (row: ArchitectureCapabilitiesRow): ArchitectureCapabilitiesRow => {
  const value: unknown = row;
  const isValid =
    isObject(value) &&
    typeof value.base === 'string' &&
    (value.variant === null || typeof value.variant === 'string') &&
    isObject(value.features) &&
    (value.defaults === null || isObject(value.defaults)) &&
    (value.vae === null || (isObject(value.vae) && Array.isArray(value.vae.accepted)));

  if (!isValid) {
    throw new TypeError(`Malformed architecture capabilities row: ${JSON.stringify(value)}`);
  }

  return row;
};

const key = (base: string, variant: string | null): string => `${base}\u0000${variant ?? ''}`;

let rows: readonly ArchitectureCapabilitiesRow[] | null = null;
let byKey = new Map<string, ArchitectureCapabilitiesRow>();
/** Every row of a base: its own and its variants'. */
let byBase = new Map<string, ArchitectureCapabilitiesRow[]>();
let revision = 0;

/** Synchronous readers observe the registry revision signal. */
const channel = createListenerChannel();

/** Subscribe to table replacements. Returns an unsubscribe function. */
export const onArchitectureCapabilitiesChanged = channel.subscribe;

/** Identity of the table currently held; `0` while there is none. */
export const getArchitectureCapabilitiesRevision = (): number => revision;

/** Called by `data/` once the table has been fetched. Throws, publishing nothing, on a malformed row. */
export const setArchitectureCapabilities = (next: readonly ArchitectureCapabilitiesRow[]): void => {
  const nextByKey = new Map(next.map((row) => [key(assertRow(row).base, row.variant), row]));
  const nextByBase = new Map<string, ArchitectureCapabilitiesRow[]>();

  for (const row of next) {
    nextByBase.set(row.base, [...(nextByBase.get(row.base) ?? []), row]);
  }

  rows = next;
  byKey = nextByKey;
  byBase = nextByBase;
  revision += 1;
  channel.notify();
};

/** Drop the table. The store calls this on account change; tests call it to isolate. */
export const resetArchitectureCapabilities = (): void => {
  if (rows === null) {
    return;
  }

  rows = null;
  byKey = new Map();
  byBase = new Map();
  revision = 0;
  channel.notify();
};

/** Hold `rows` as the table while `run` reads it, then drop it: for offline tooling such as graph export. */
export const withArchitectureCapabilities = <T>(next: readonly ArchitectureCapabilitiesRow[], run: () => T): T => {
  setArchitectureCapabilities(next);

  try {
    return run();
  } finally {
    resetArchitectureCapabilities();
  }
};

export const hasArchitectureCapabilities = (): boolean => rows !== null;

/** `(base, variant)` if that variant answers differently, else the architecture's own row. */
export const getArchitectureCapabilityRow = (
  base: string,
  variant?: unknown
): ArchitectureCapabilitiesRow | undefined => {
  if (typeof variant === 'string' && variant.length > 0) {
    const exact = byKey.get(key(base, variant));
    if (exact) {
      return exact;
    }
  }
  return byKey.get(key(base, null));
};

/** Look up the variant first; omitting it uses the base row. */
export const getArchitectureFeatures = (
  base: string,
  variant?: unknown
): ArchitectureCapabilitiesRow['features'] | undefined => getArchitectureCapabilityRow(base, variant)?.features;

type VaeCandidate = { base: string; latent_channels?: number | null };

const rowAcceptsVae = (row: ArchitectureCapabilitiesRow, vae: VaeCandidate): boolean => {
  // A null `vae` block is the backend saying "its own base, no constraints" — what `accepts_vae`
  // answers for an architecture that declares no facet. It is a declaration, not a missing one.
  if (!row.vae) {
    return vae.base === row.base;
  }

  return row.vae.accepted.some(
    (accepted) =>
      accepted.base === vae.base &&
      (accepted.latent_channels === null || vae.latent_channels === accepted.latent_channels)
  );
};

/**
 * Whether a model of `host`'s architecture decodes with `vae`: the backend's `accepts_vae`. Without the table nothing
 * is accepted. Pass the host's variant where it has one: Wan A14B and TI2V-5B decode with different VAEs, and only
 * the variant row says so.
 */
export const architectureAcceptsVae = (host: { base: string; variant?: unknown }, vae: VaeCandidate): boolean => {
  const row = getArchitectureCapabilityRow(host.base, host.variant);

  return row !== undefined && rowAcceptsVae(row, vae);
};

/**
 * Whether some variant of `base` decodes with `vae`, its own row included: the backend's variant-agnostic
 * `accepted_vae_bases`, with latent widths kept per row. For models that name no main variant, such as LoRAs, whose
 * variants are not the main's. Without the table nothing is accepted.
 */
export const someArchitectureVariantAcceptsVae = (base: string, vae: VaeCandidate): boolean =>
  (byBase.get(base) ?? []).some((row) => rowAcceptsVae(row, vae));
