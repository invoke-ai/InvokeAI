import type { GenerateComponentValueKey } from './baseGenerationPolicies';

/** A component slot a main model can store a default model for. */
export interface DefaultComponentSlot {
  /** The settings key the picker fills. */
  setting: Exclude<GenerateComponentValueKey, 'vae'>;
  /**
   * The slot's name in `default_settings.components` and the metadata key a generation records it under; one
   * canonical name per slot, `^[a-z0-9_]+$` like the backend requires. The VAE has its own `default_settings.vae`.
   */
  persistenceKey: string;
  /** The slot holds a main model (a component source or second transformer), not an encoder. */
  isMainModel?: true;
}

/**
 * The one table pairing slots with their persistence key. Recall reads the same keys from image metadata, so the
 * stored defaults and the recorded components cannot drift apart.
 */
export const DEFAULT_COMPONENT_SLOTS: readonly DefaultComponentSlot[] = [
  { persistenceKey: 'anima_inpaint_adapter', setting: 'animaInpaintAdapterModel' },
  { persistenceKey: 'clip_embed_model', setting: 'clipEmbedModel' },
  { persistenceKey: 'clip_g_model', setting: 'clipGEmbedModel' },
  { persistenceKey: 'clip_l_model', setting: 'clipLEmbedModel' },
  { isMainModel: true, persistenceKey: 'component_source', setting: 'componentSourceModel' },
  { persistenceKey: 'flux3_prompt_upsampler', setting: 'flux3PromptUpsamplerModel' },
  { persistenceKey: 'gemma2_encoder', setting: 'gemma2EncoderModel' },
  { isMainModel: true, persistenceKey: 'ideogram4_unconditional_model', setting: 'ideogram4UnconditionalModel' },
  { persistenceKey: 'mistral_encoder', setting: 'mistralEncoderModel' },
  { persistenceKey: 'pid_decoder', setting: 'pidDecoderModel' },
  { persistenceKey: 'qwen3_5_encoder', setting: 'qwen35EncoderModel' },
  { persistenceKey: 'qwen3_encoder', setting: 'qwen3EncoderModel' },
  { persistenceKey: 'qwen3_vl_encoder', setting: 'qwen3VLEncoderModel' },
  { persistenceKey: 'qwen3_vl_vision_tower', setting: 'qwen3VLVisionTowerModel' },
  { persistenceKey: 'qwen_image_qwen_vl_encoder', setting: 'qwenVLEncoderModel' },
  { persistenceKey: 't5_encoder', setting: 't5EncoderModel' },
  { isMainModel: true, persistenceKey: 'wan_transformer_low_noise', setting: 'wanLowNoiseModel' },
  { persistenceKey: 'wan_t5_encoder_model', setting: 'wanT5EncoderModel' },
];

export const getDefaultComponentSlot = (persistenceKey: string): DefaultComponentSlot | undefined =>
  DEFAULT_COMPONENT_SLOTS.find((slot) => slot.persistenceKey === persistenceKey);
