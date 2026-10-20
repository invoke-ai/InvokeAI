import type { ModelIdentifierField } from 'features/nodes/types/common';

type AnimaComponentSyncArg = {
  selectedVae: ModelIdentifierField | null;
  selectedEncoder: ModelIdentifierField | null;
  /** Anima-base VAEs (Wan 2.1 / QwenImage) - what the model was trained with. */
  nativeVaes: ModelIdentifierField[];
  /** Everything the Anima loader accepts, i.e. the native ones plus FLUX and 16-channel Wan VAEs. */
  compatibleVaes: ModelIdentifierField[];
  /** Qwen3 0.6B encoders - the only variant whose 1024-wide embeddings Anima can consume. */
  availableEncoders: ModelIdentifierField[];
};

type AnimaComponentUpdates = {
  vae?: ModelIdentifierField | null;
  encoder?: ModelIdentifierField | null;
};

/**
 * Reconcile the Anima standalone-component slots against the installed models.
 *
 * Anima keeps its VAE and Qwen3 encoder in dedicated params slots, and nothing validated them when the
 * model list changed: uninstalling the selected VAE left a dangling key in state that only surfaced as
 * a failed generation. Both slots are required by `buildAnimaGraph`, so a dangling one is not merely
 * cosmetic.
 *
 * Returns only the keys that need dispatching, so an unchanged slot causes no dispatch at all.
 */
export const getAnimaComponentUpdates = (arg: AnimaComponentSyncArg): AnimaComponentUpdates => {
  const { selectedVae, selectedEncoder, nativeVaes, compatibleVaes, availableEncoders } = arg;

  // Same preference as the modelSelected listener: a native Anima VAE beats a FLUX/Wan fallback, which
  // the loader merely tolerates and which decodes on a different code path.
  const defaultVae = nativeVaes[0] ?? compatibleVaes[0];
  const defaultEncoder = availableEncoders[0];

  const hasSelectedVae = selectedVae !== null && selectedVae !== undefined;
  const hasSelectedEncoder = selectedEncoder !== null && selectedEncoder !== undefined;
  const selectedVaeIsAvailable = hasSelectedVae && compatibleVaes.some((vae) => vae.key === selectedVae.key);
  const selectedEncoderIsAvailable =
    hasSelectedEncoder && availableEncoders.some((encoder) => encoder.key === selectedEncoder.key);

  return {
    ...(!selectedVaeIsAvailable && (hasSelectedVae || defaultVae) ? { vae: defaultVae ?? null } : {}),
    ...(!selectedEncoderIsAvailable && (hasSelectedEncoder || defaultEncoder)
      ? { encoder: defaultEncoder ?? null }
      : {}),
  };
};
