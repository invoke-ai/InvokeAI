import type { HoverCard } from '@chakra-ui/react';

type HintPlacement = NonNullable<NonNullable<HoverCard.RootProps['positioning']>['placement']>;

interface HintDefinition {
  /** Support article opened by the card's "Learn more" link. */
  href?: string;
  /** Override DEFAULT_HINT_PLACEMENT when the card would cover its own control. */
  placement?: HintPlacement;
}

/** Cards sit beside the panel rather than over the control they explain. */
export const DEFAULT_HINT_PLACEMENT: HintPlacement = 'right';

const SUPPORT_ADVANCED_SETTINGS = 'https://support.invoke.ai/support/solutions/articles/151000178161-advanced-settings';
const SUPPORT_CONCEPTS = 'https://support.invoke.ai/support/solutions/articles/151000159072';

/** Keep prose in hints.* and non-translatable metadata here; tests enforce registry/catalog parity. */
export const FEATURE_HINTS = {
  aspectRatio: {},
  cfgRescale: { href: SUPPORT_ADVANCED_SETTINGS },
  cfgScale: { href: 'https://www.youtube.com/watch?v=1OeHEJrsTpI' },
  clipEmbed: {},
  clipSkip: { href: SUPPORT_ADVANCED_SETTINGS },
  colorCompensation: {},
  concepts: { href: SUPPORT_CONCEPTS },
  conditioningRebalance: {},
  creativity: {},
  guidance: {},
  height: {},
  hidiffusion: { href: 'https://github.com/megvii-research/HiDiffusion' },
  hidiffusionRauNet: { href: 'https://github.com/megvii-research/HiDiffusion' },
  hidiffusionT1Ratio: { href: 'https://github.com/megvii-research/HiDiffusion' },
  hidiffusionT2Ratio: { href: 'https://github.com/megvii-research/HiDiffusion' },
  hidiffusionWindowAttn: { href: 'https://github.com/megvii-research/HiDiffusion' },
  imageInfluence: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000094998-image-to-image',
  },
  layerStackControl: {},
  layerStackInpaintMask: {},
  layerStackRaster: {},
  layerStackRegionalGuidance: {},
  model: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000096601-what-is-a-model-which-should-i-use-',
  },
  negativePrompt: {},
  pidMode: { href: 'https://github.com/nv-tlabs/PiD' },
  positivePrompt: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000096606-tips-on-crafting-prompts',
  },
  referenceImage: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000159340-global-and-regional-reference-images-ip-adapters-',
  },
  referenceImageWeight: {},
  scheduler: { href: 'https://www.youtube.com/watch?v=1OeHEJrsTpI' },
  seamlessTiling: { href: SUPPORT_ADVANCED_SETTINGS },
  seed: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000096684-what-is-a-seed-how-do-i-use-it-to-recreate-the-same-image-',
  },
  steps: {},
  structure: {},
  t5Encoder: {},
  tileControlNet: { href: 'https://support.invoke.ai/support/solutions/articles/151000105880' },
  tileOverlap: {},
  tileSize: {},
  upscaleModel: {},
  upscaleScale: {},
  vae: { href: SUPPORT_ADVANCED_SETTINGS },
  vaePrecision: { href: SUPPORT_ADVANCED_SETTINGS },
  width: {},
} as const satisfies Record<string, HintDefinition>;

export type FeatureHintId = keyof typeof FEATURE_HINTS;

export const getFeatureHint = (id: FeatureHintId): HintDefinition => FEATURE_HINTS[id];
