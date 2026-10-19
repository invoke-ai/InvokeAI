import type { AnyModelConfig } from 'services/api/types';

/**
 * Wan 2.2 single-file mains are transformer-only: the VAE and UMT5-XXL encoder have to
 * come from standalone models or from a Diffusers "Component Source". This computes
 * which of those slots need to change when the main model is selected.
 *
 * It both fills empty slots and re-points stale ones. Nothing else clears them —
 * `paramsSlice` carries all four across a base change and `modelsLoaded` has no Wan
 * handler — so a slot auto-filled for a previous main survives into the next one, where
 * the loader validates it against the new variant and refuses.
 *
 * The variant matters because A14B (t2v_a14b / i2v_a14b) and TI2V-5B use different VAEs:
 * 16-channel Wan 2.1 vs 48-channel Wan 2.2. `WanModelLoaderInvocation._validate_standalone_vae`
 * and `_validate_component_source_vae` reject a mismatch outright.
 */

type Identifier = { key: string } | null;

type WanComponentUpdates = {
  /** Present only when the slot should change. `null` means clear it. */
  vae?: AnyModelConfig | null;
  componentSource?: AnyModelConfig | null;
  encoder?: AnyModelConfig | null;
  lowNoisePartner?: AnyModelConfig | null;
};

const variantOf = (model: unknown): string | null =>
  model && typeof model === 'object' && 'variant' in model && typeof model.variant === 'string' ? model.variant : null;

/** Exported so the readiness pre-flight can gate on the same single-expert test the
 *  loader uses, instead of restating `variant === 'ti2v_5b'` in a second place. */
export const isWanTi2v5b = (model: unknown): boolean => variantOf(model) === 'ti2v_5b';

/** Mirrors `WanModelLoaderInvocation._validate_standalone_vae`: TI2V-5B needs the
 *  48-channel Wan 2.2 VAE, A14B the 16-channel Wan 2.1 one. Exported so the readiness
 *  pre-flight tests the same rule rather than a second, drifting copy of it. */
export const isWanVaeCompatible = (mainConfig: unknown, vae: unknown): boolean =>
  !!vae &&
  typeof vae === 'object' &&
  'latent_channels' in vae &&
  vae.latent_channels === (isWanTi2v5b(mainConfig) ? 48 : 16);

/** Mirrors `_validate_component_source_vae` plus `_validate_component_source_format`:
 *  the source must be a Diffusers Wan main on the same side of the TI2V-5B / A14B split,
 *  because its VAE is what gets used. */
export const isWanComponentSourceCompatible = (mainConfig: unknown, source: unknown): boolean =>
  !!source &&
  typeof source === 'object' &&
  'format' in source &&
  source.format === 'diffusers' &&
  isWanTi2v5b(source) === isWanTi2v5b(mainConfig);

/** The low-noise partner must match the main's variant exactly — a stricter rule than the
 *  TI2V/A14B split above. See `wan_model_loader.py`'s "must use the same Wan variant". */
export const isWanLowNoisePartnerCompatible = (mainConfig: unknown, partner: unknown): boolean =>
  variantOf(partner) === variantOf(mainConfig);

export const getWanComponentUpdates = (arg: {
  /** The newly selected Wan main model's config. */
  mainConfig: AnyModelConfig;
  /** True for GGUF / safetensors-checkpoint mains; false for Diffusers. */
  isSingleFileMain: boolean;
  /**
   * Configs of the currently wired slots, resolved against the installed models —
   * `null` when the slot is empty *or* points at a model that no longer exists, which
   * are handled the same way.
   */
  selectedVae: AnyModelConfig | null;
  selectedComponentSource: AnyModelConfig | null;
  selectedEncoder: Identifier;
  selectedLowNoisePartner: AnyModelConfig | null;
  availableVaes: AnyModelConfig[];
  availableDiffusers: AnyModelConfig[];
  availableEncoders: AnyModelConfig[];
}): WanComponentUpdates => {
  const {
    mainConfig,
    isSingleFileMain,
    selectedVae,
    selectedComponentSource,
    selectedEncoder,
    selectedLowNoisePartner,
    availableVaes,
    availableDiffusers,
    availableEncoders,
  } = arg;

  const updates: WanComponentUpdates = {};

  const vaeIsCompatible = (model: unknown) => isWanVaeCompatible(mainConfig, model);
  const sourceIsCompatible = (model: unknown) => isWanComponentSourceCompatible(mainConfig, model);

  // A wired standalone VAE outranks every other source in the loader — including a
  // Diffusers main's own — so an incompatible one has to be corrected for *any* Wan main.
  // Filling an empty slot is different: only a single-file main needs one. Auto-wiring a
  // standalone VAE for a self-contained Diffusers main would silently override the VAE it
  // ships with, and the user could not undo it (clearing the combobox would just refill
  // on the next selection).
  if (selectedVae && !vaeIsCompatible(selectedVae)) {
    updates.vae = availableVaes.find(vaeIsCompatible) ?? null;
  } else if (!selectedVae && isSingleFileMain) {
    const vae = availableVaes.find(vaeIsCompatible);
    if (vae) {
      updates.vae = vae;
    }
  }

  // The Component Source only feeds a single-file main; a Diffusers main carries its own
  // components and the loader never consults it for the VAE.
  if (isSingleFileMain) {
    if (!selectedComponentSource || !sourceIsCompatible(selectedComponentSource)) {
      // No "any Wan Diffusers model" fallback. Picking an arbitrary one here produces
      // exactly the mismatch the loader validation exists to catch. Clearing when nothing
      // fits is deliberate: an empty slot reads as "pick one" in the UI, a stale one reads
      // as already handled.
      const source = availableDiffusers.find(sourceIsCompatible);
      if (source) {
        updates.componentSource = source;
      } else if (selectedComponentSource) {
        updates.componentSource = null;
      }
    }

    // The UMT5-XXL encoder is shared across every Wan variant, so first-match is correct
    // and there is nothing to re-validate beyond the model still existing.
    if (!selectedEncoder) {
      const encoder = availableEncoders[0];
      if (encoder) {
        updates.encoder = encoder;
      }
    }
  }

  // The low-noise partner. Its check is exact variant equality, stricter than the VAE's
  // TI2V/A14B split, so switching t2v -> i2v leaves a partner the loader will refuse.
  // There is no safe auto-repoint here — which file is the partner is the user's call —
  // so an incompatible one is cleared and the slot goes back to reading "pick one".
  if (selectedLowNoisePartner && !isWanLowNoisePartnerCompatible(mainConfig, selectedLowNoisePartner)) {
    updates.lowNoisePartner = null;
  }

  return updates;
};
