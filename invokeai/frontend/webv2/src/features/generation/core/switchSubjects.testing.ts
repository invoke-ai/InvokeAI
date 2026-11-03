import type { GenerateComponentValueKey, GenerateModelSwitchSubject } from './baseGenerationPolicies';

import { GENERATE_COMPONENT_VALUE_KEYS } from './baseGenerationPolicies';

/** The subjects a switch can report that are not component slots; a Record so a new subject cannot be left out. */
const SWITCH_SETTING_SUBJECTS: Record<Exclude<GenerateModelSwitchSubject, GenerateComponentValueKey>, true> = {
  cfgRescale: true,
  clipSkip: true,
  colorCompensation: true,
  concepts: true,
  guidance: true,
  hiDiffusion: true,
  referenceImages: true,
  seamless: true,
  vaePrecision: true,
};

/** Every subject a switch can report; the locale names each under `widgets.generate.switchSubjects`. */
export const GENERATE_MODEL_SWITCH_SUBJECTS = [
  ...GENERATE_COMPONENT_VALUE_KEYS,
  ...Object.keys(SWITCH_SETTING_SUBJECTS),
] as readonly GenerateModelSwitchSubject[];
