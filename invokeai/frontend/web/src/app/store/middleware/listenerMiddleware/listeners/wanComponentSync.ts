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
};

const variantOf = (model: unknown): string | null =>
  model && typeof model === 'object' && 'variant' in model && typeof model.variant === 'string' ? model.variant : null;

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
    availableVaes,
    availableDiffusers,
    availableEncoders,
  } = arg;

  const updates: WanComponentUpdates = {};

  const isTi2v5b = variantOf(mainConfig) === 'ti2v_5b';
  const requiredLatentChannels = isTi2v5b ? 48 : 16;

  const vaeIsCompatible = (model: unknown) =>
    !!model &&
    typeof model === 'object' &&
    'latent_channels' in model &&
    model.latent_channels === requiredLatentChannels;

  const sourceIsCompatible = (model: unknown) => (variantOf(model) === 'ti2v_5b') === isTi2v5b;

  // The standalone VAE outranks every other source in the loader — including a Diffusers
  // main's own — so it is checked for any Wan main, not just single-file ones.
  if (!vaeIsCompatible(selectedVae)) {
    const vae = availableVaes.find(vaeIsCompatible);
    // Clearing when nothing fits is deliberate: an empty slot reads as "pick one" in the
    // UI, a stale one reads as already handled.
    if (vae) {
      updates.vae = vae;
    } else if (selectedVae) {
      updates.vae = null;
    }
  }

  // The Component Source only feeds a single-file main; a Diffusers main carries its own
  // components and the loader never consults it for the VAE.
  if (isSingleFileMain) {
    if (!selectedComponentSource || !sourceIsCompatible(selectedComponentSource)) {
      // No "any Wan Diffusers model" fallback. Picking an arbitrary one here produces
      // exactly the mismatch the loader validation exists to catch.
      const source = availableDiffusers.find(sourceIsCompatible);
      if (source) {
        updates.componentSource = source;
      } else if (selectedComponentSource) {
        updates.componentSource = null;
      }
    }

    // The UMT5-XXL encoder is shared across every Wan variant, so first-match is correct
    // and there is nothing to re-validate.
    if (!selectedEncoder) {
      const encoder = availableEncoders[0];
      if (encoder) {
        updates.encoder = encoder;
      }
    }
  }

  return updates;
};
