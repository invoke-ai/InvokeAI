import type { PopoverProps } from '@invoke-ai/ui-library';
import commercialLicenseBg from 'public/assets/images/commercial-license-bg.png';
import denoisingStrength from 'public/assets/images/denoising-strength.png';

export type Feature =
  | 'clipSkip'
  | 'hrf'
  | 'paramNegativeConditioning'
  | 'paramPositiveConditioning'
  | 'paramScheduler'
  | 'compositingMaskBlur'
  | 'compositingBlurMethod'
  | 'compositingCoherencePass'
  | 'compositingCoherenceMode'
  | 'compositingCoherenceEdgeSize'
  | 'compositingCoherenceMinDenoise'
  | 'compositingMaskAdjustments'
  | 'controlNet'
  | 'controlNetBeginEnd'
  | 'controlNetControlMode'
  | 'controlNetProcessor'
  | 'controlNetResizeMode'
  | 'controlNetWeight'
  | 'dynamicPrompts'
  | 'dynamicPromptsMaxPrompts'
  | 'dynamicPromptsSeedBehaviour'
  | 'globalReferenceImage'
  | 'imageFit'
  | 'infillMethod'
  | 'inpainting'
  | 'ipAdapterMethod'
  | 'lora'
  | 'loraWeight'
  | 'noiseUseCPU'
  | 'paramAspect'
  | 'paramCFGScale'
  | 'paramGuidance'
  | 'paramCFGRescaleMultiplier'
  | 'paramDenoisingStrength'
  | 'paramHeight'
  | 'paramHrf'
  | 'paramIterations'
  | 'paramModel'
  | 'paramRatio'
  | 'paramSeed'
  | 'paramSteps'
  | 'paramUpscaleMethod'
  | 'paramVAE'
  | 'paramVAEPrecision'
  | 'paramWidth'
  | 'patchmatchDownScaleSize'
  | 'rasterLayer'
  | 'refinerModel'
  | 'refinerNegativeAestheticScore'
  | 'refinerPositiveAestheticScore'
  | 'refinerScheduler'
  | 'refinerStart'
  | 'refinerSteps'
  | 'refinerCfgScale'
  | 'regionalGuidance'
  | 'regionalGuidanceAndReferenceImage'
  | 'regionalReferenceImage'
  | 'scaleBeforeProcessing'
  | 'seamlessTilingXAxis'
  | 'seamlessTilingYAxis'
  | 'upscaleModel'
  | 'scale'
  | 'creativity'
  | 'structure'
  | 'optimizedDenoising'
  | 'fluxDevLicense';

export type PopoverData = PopoverProps & {
  image?: string;
  href?: string;
  buttonLabel?: string;
};

export const POPOVER_DATA: { [key in Feature]?: PopoverData } = {
  paramNegativeConditioning: {
    placement: 'right',
  },
  clipSkip: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000178161-advanced-settings',
  },
  inpainting: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000096702-inpainting-outpainting-and-bounding-box',
  },
  rasterLayer: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000094998-raster-layers-and-initial-images',
  },
  regionalGuidance: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000165024-regional-guidance-layers',
  },
  regionalGuidanceAndReferenceImage: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000165024-regional-guidance-layers',
  },
  globalReferenceImage: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000159340-global-and-regional-reference-images-ip-adapters-',
  },
  regionalReferenceImage: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000159340-global-and-regional-reference-images-ip-adapters-',
  },
  controlNet: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000105880',
  },
  controlNetBeginEnd: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000178148',
  },
  controlNetWeight: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000178148',
  },
  lora: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000159072',
  },
  loraWeight: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000159072-concepts-low-rank-adaptations-loras-',
  },
  compositingMaskBlur: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000158838-compositing-settings',
  },
  compositingBlurMethod: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000158838-compositing-settings',
  },
  compositingCoherenceMode: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000158838-compositing-settings',
  },
  infillMethod: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000158838-compositing-settings',
  },
  scaleBeforeProcessing: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000179777-scale-before-processing',
  },
  paramCFGScale: {
    href: 'https://www.youtube.com/watch?v=1OeHEJrsTpI',
  },
  paramCFGRescaleMultiplier: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000178161-advanced-settings',
  },
  paramDenoisingStrength: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000094998-image-to-image',
    image: denoisingStrength,
  },
  paramHrf: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000096700-how-can-i-get-larger-images-what-does-upscaling-do-',
  },
  paramIterations: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000159073',
  },
  paramPositiveConditioning: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000096606-tips-on-crafting-prompts',
    placement: 'right',
  },
  paramScheduler: {
    placement: 'right',
    href: 'https://www.youtube.com/watch?v=1OeHEJrsTpI',
  },
  paramSeed: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000096684-what-is-a-seed-how-do-i-use-it-to-recreate-the-same-image-',
  },
  paramModel: {
    placement: 'right',
    href: 'https://support.invoke.ai/support/solutions/articles/151000096601-what-is-a-model-which-should-i-use-',
  },
  paramRatio: {
    gutter: 16,
  },
  controlNetControlMode: {
    placement: 'right',
    href: 'https://support.invoke.ai/support/solutions/articles/151000178148',
  },
  controlNetProcessor: {
    placement: 'right',
    href: 'https://support.invoke.ai/support/solutions/articles/151000105880-using-controlnet',
  },
  controlNetResizeMode: {
    placement: 'right',
    href: 'https://support.invoke.ai/support/solutions/articles/151000178148',
  },
  paramVAE: {
    placement: 'right',
    href: 'https://support.invoke.ai/support/solutions/articles/151000178161-advanced-settings',
  },
  paramVAEPrecision: {
    placement: 'right',
    href: 'https://support.invoke.ai/support/solutions/articles/151000178161-advanced-settings',
  },
  paramUpscaleMethod: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000096700-how-can-i-get-larger-images-what-does-upscaling-do-',
  },
  refinerModel: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000178333-using-the-refiner',
  },
  refinerNegativeAestheticScore: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000178333-using-the-refiner',
  },
  refinerPositiveAestheticScore: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000178333-using-the-refiner',
  },
  refinerScheduler: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000178333-using-the-refiner',
  },
  refinerStart: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000178333-using-the-refiner',
  },
  refinerSteps: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000178333-using-the-refiner',
  },
  refinerCfgScale: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000178333-using-the-refiner',
  },
  seamlessTilingXAxis: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000178161-advanced-settings',
  },
  seamlessTilingYAxis: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000178161-advanced-settings',
  },
  fluxDevLicense: {
    href: 'https://www.invoke.com/get-a-commercial-license-for-flux',
    image: commercialLicenseBg,
  },
} as const;

export const OPEN_DELAY = 1000; // in milliseconds

export const POPPER_MODIFIERS: PopoverProps['modifiers'] = [{ name: 'preventOverflow', options: { padding: 10 } }];
