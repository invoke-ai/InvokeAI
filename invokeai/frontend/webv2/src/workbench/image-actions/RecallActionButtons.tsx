import {
  AsteriskIcon,
  QuoteIcon,
  RulerIcon,
  ScissorsIcon,
  ShuffleIcon,
  SproutIcon,
  type LucideIcon,
} from 'lucide-react';

import type { ImageRecallCapabilities, ImageRecallKind } from './imageRecall';

/**
 * Share recall labels/icons across metadata and queue details. Unavailable actions stay visible but disabled with
 * the host's reason.
 */

const RECALL_ACTION_ITEMS: {
  capability: keyof ImageRecallCapabilities;
  icon: LucideIcon;
  kind: ImageRecallKind;
  label: string;
}[] = [
  { capability: 'all', icon: AsteriskIcon, kind: 'all', label: 'Recall All' },
  { capability: 'remix', icon: ShuffleIcon, kind: 'remix', label: 'Remix Image' },
  { capability: 'prompts', icon: QuoteIcon, kind: 'prompts', label: 'Use Prompt' },
  { capability: 'seed', icon: SproutIcon, kind: 'seed', label: 'Use Seed' },
  { capability: 'dimensions', icon: RulerIcon, kind: 'dimensions', label: 'Use Size' },
  { capability: 'clipSkip', icon: ScissorsIcon, kind: 'clipSkip', label: 'Use CLIP Skip' },
];

/** The recall verbs in their canonical order, for hosts that lay them out themselves. */
export const IMAGE_RECALL_KINDS: readonly ImageRecallKind[] = RECALL_ACTION_ITEMS.map((item) => item.kind);

/** Reuse recall icons and labels in controls outside the shared row. */
export const getImageRecallVerb = (kind: ImageRecallKind): { icon: LucideIcon; label: string } => {
  const item = RECALL_ACTION_ITEMS.find((candidate) => candidate.kind === kind) ?? RECALL_ACTION_ITEMS[0]!;

  return { icon: item.icon, label: item.label };
};
