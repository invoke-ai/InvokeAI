import type { PopoverProps } from '@invoke-ai/ui-library';

export type Feature =
  | 'clipSkip'
  | 'hrf'
  | 'paramNegativeConditioning'
  | 'paramPositiveConditioning'
  | 'paramScheduler'
  | 'compositingBlur'
  | 'compositingBlurMethod'
  | 'compositingCoherencePass'
  | 'compositingCoherenceMode'
  | 'compositingCoherenceSteps'
  | 'compositingStrength'
  | 'compositingMaskAdjustments'
  | 'controlNetBeginEnd'
  | 'controlNetControlMode'
  | 'controlNetResizeMode'
  | 'controlNet'
  | 'controlNetWeight'
  | 'dynamicPrompts'
  | 'dynamicPromptsMaxPrompts'
  | 'dynamicPromptsSeedBehaviour'
  | 'infillMethod'
  | 'lora'
  | 'noiseUseCPU'
  | 'paramCFGScale'
  | 'paramCFGRescaleMultiplier'
  | 'paramDenoisingStrength'
  | 'paramIterations'
  | 'paramModel'
  | 'paramRatio'
  | 'paramSeed'
  | 'paramSteps'
  | 'paramVAE'
  | 'paramVAEPrecision'
  | 'scaleBeforeProcessing';

export type PopoverData = PopoverProps & {
  image?: string;
  href?: string;
  buttonLabel?: string;
};

export const POPOVER_DATA: { [key in Feature]?: PopoverData } = {
  paramNegativeConditioning: {
    placement: 'right',
  },
  controlNet: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000105880',
  },
  lora: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000159072',
  },
  compositingCoherenceMode: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000158838',
  },
  infillMethod: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000158841',
  },
  scaleBeforeProcessing: {
    href: 'https://support.invoke.ai/support/solutions/articles/151000158841',
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
    href: 'https://support.invoke.ai/support/solutions/articles/151000159073',
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
  },
  controlNetResizeMode: {
    placement: 'right',
  },
  paramVAE: {
    placement: 'right',
  },
  paramVAEPrecision: {
    placement: 'right',
  },
} as const;

export const OPEN_DELAY = 1000; // in milliseconds

export const POPPER_MODIFIERS: PopoverProps['modifiers'] = [{ name: 'preventOverflow', options: { padding: 10 } }];
