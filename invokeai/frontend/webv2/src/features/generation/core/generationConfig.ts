/** Shared types avoid a mapper/policy circular dependency. */

export type SchedulerSetId = 'standard' | 'flow' | 'flow-no-lcm' | 'anima';

export type NegativePromptUsage = 'always' | 'cfg-gated' | 'never';

export type GuidanceLabel = 'CFG' | 'Guidance';

export interface BaseGenerationConfig {
  dimensions: {
    grid: number;
    /** optimal is the square root of pixel area, not source width. */
    optimalSide: number;
  };
  defaults: {
    steps: number;
    /** Whatever the single guidance slider shows -- CFG or distilled guidance, per `guidanceLabel`. */
    cfgScale: number;
    scheduler: string;
  };
  schedulerSet: SchedulerSetId;
  schedulerAppliesToGraph: boolean;
  guidanceLabel: GuidanceLabel;
  /** Denoise validation bounds differ from the slider track; max=null means unbounded. */
  guidance: {
    min: number;
    max: number | null;
  };
  negativePrompt: {
    visible: boolean;
    usage: NegativePromptUsage;
  };
  ui: {
    sdVaeOverride: boolean;
    colorCompensation: boolean;
    vaePrecision: boolean;
    seamless: boolean;
    cfgRescale: boolean;
    clipSkipMax?: number;
  };
}
