import type { TFunction } from 'i18next';

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
 * Shared recall labels/icons for metadata and queue details; unavailable actions stay visible, disabled with a reason.
 */

const RECALL_ACTION_ITEMS = [
  { capability: 'all', icon: AsteriskIcon, kind: 'all' },
  { capability: 'remix', icon: ShuffleIcon, kind: 'remix' },
  { capability: 'prompts', icon: QuoteIcon, kind: 'prompts' },
  { capability: 'seed', icon: SproutIcon, kind: 'seed' },
  { capability: 'dimensions', icon: RulerIcon, kind: 'dimensions' },
  { capability: 'clipSkip', icon: ScissorsIcon, kind: 'clipSkip' },
] as const satisfies readonly {
  capability: keyof ImageRecallCapabilities;
  icon: LucideIcon;
  kind: ImageRecallKind;
}[];

/** The recall verbs in their canonical order, for hosts that lay them out themselves. */
export const IMAGE_RECALL_KINDS = RECALL_ACTION_ITEMS.map((item) => item.kind);

/** Reuse recall icons and labels in controls outside the shared row. */
export const getImageRecallVerb = (kind: ImageRecallKind, t: TFunction): { icon: LucideIcon; label: string } => {
  if (kind === 'positivePrompt' || kind === 'negativePrompt') {
    return { icon: QuoteIcon, label: t(`widgets.gallery.itemActions.recall.verbs.${kind}`) };
  }

  const item = RECALL_ACTION_ITEMS.find((candidate) => candidate.kind === kind) ?? RECALL_ACTION_ITEMS[0]!;

  return { icon: item.icon, label: t(`widgets.gallery.itemActions.recall.verbs.${item.kind}`) };
};
