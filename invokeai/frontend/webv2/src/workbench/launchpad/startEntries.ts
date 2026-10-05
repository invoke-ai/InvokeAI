import type { LaunchpadIntentId } from '@workbench/launchpad/intents';
import type { LucideIcon } from 'lucide-react';

import { LAUNCHPAD_INTENT_IDS } from '@workbench/launchpad/intents';
import { BlocksIcon, BrushIcon, ClapperboardIcon, ScalingIcon, TypeIcon } from 'lucide-react';

/** One presentation per intent so the Home tiles and the new-project menu offer the same starts. */
export interface LaunchpadStartEntry {
  id: LaunchpadIntentId;
  icon: LucideIcon;
  labelKey: string;
  descriptionKey: string;
  search: { intent: LaunchpadIntentId; new: true };
}

const START_ICONS: Record<LaunchpadIntentId, LucideIcon> = {
  canvas: BrushIcon,
  generate: TypeIcon,
  upscale: ScalingIcon,
  video: ClapperboardIcon,
  workflow: BlocksIcon,
};

// Spelled out rather than interpolated, so the translation-key report can see them.
const START_LABEL_KEYS: Record<LaunchpadIntentId, string> = {
  canvas: 'launchpad.home.intents.canvas',
  generate: 'launchpad.home.intents.generate',
  upscale: 'launchpad.home.intents.upscale',
  video: 'launchpad.home.intents.video',
  workflow: 'launchpad.home.intents.workflow',
};

const START_DESCRIPTION_KEYS: Record<LaunchpadIntentId, string> = {
  canvas: 'launchpad.home.intents.canvasDescription',
  generate: 'launchpad.home.intents.generateDescription',
  upscale: 'launchpad.home.intents.upscaleDescription',
  video: 'launchpad.home.intents.videoDescription',
  workflow: 'launchpad.home.intents.workflowDescription',
};

export const LAUNCHPAD_START_ENTRIES: readonly LaunchpadStartEntry[] = LAUNCHPAD_INTENT_IDS.map((id) => ({
  descriptionKey: START_DESCRIPTION_KEYS[id],
  icon: START_ICONS[id],
  id,
  labelKey: START_LABEL_KEYS[id],
  search: { intent: id, new: true },
}));
