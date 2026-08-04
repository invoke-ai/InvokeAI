import type { HoverCard } from '@chakra-ui/react';

type HintPlacement = NonNullable<NonNullable<HoverCard.RootProps['positioning']>['placement']>;

interface HintDefinition {
  /** Support article opened by the card's "Learn more" link. */
  href?: string;
  /**
   * Overrides {@link DEFAULT_HINT_PLACEMENT}. Nothing needs it today — the
   * popper flips on its own when a panel is docked against the far edge — but
   * a control whose card would cover its own value can pin itself here.
   */
  placement?: HintPlacement;
}

/** Cards sit beside the panel rather than over the control they explain. */
export const DEFAULT_HINT_PLACEMENT: HintPlacement = 'right';

const SUPPORT_ADVANCED_SETTINGS = 'https://support.invoke.ai/support/solutions/articles/151000178161-advanced-settings';
const SUPPORT_CONCEPTS = 'https://support.invoke.ai/support/solutions/articles/151000159072';

/**
 * Every hint the workbench can attach to a control. Prose lives in the `hints.*`
 * i18n namespace — this table carries only what a translator should not own.
 * `hintRegistry.test.ts` proves the two stay in step in both directions.
 */
export const FEATURE_HINTS = {
  aspectRatio: {},
  batchCount: { href: 'https://support.invoke.ai/support/solutions/articles/151000159073' },
  cfgRescale: { href: SUPPORT_ADVANCED_SETTINGS },
  cfgScale: { href: 'https://www.youtube.com/watch?v=1OeHEJrsTpI' },
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
