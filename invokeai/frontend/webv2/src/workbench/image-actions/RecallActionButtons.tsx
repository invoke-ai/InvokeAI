import { ButtonGroup, Icon } from '@chakra-ui/react';
import { Button, Tooltip } from '@platform/ui';
import {
  AsteriskIcon,
  QuoteIcon,
  RulerIcon,
  ScissorsIcon,
  ShuffleIcon,
  SproutIcon,
  type LucideIcon,
} from 'lucide-react';
import { useCallback, type ReactNode } from 'react';

import type { ImageRecallCapabilities, ImageRecallKind } from './imageRecall';

/**
 * The shared recall-verbs row: one look and one vocabulary for every surface
 * that recalls generation settings (preview metadata panel, queue item
 * details). Verbs mirror the image context menu — same labels, same icons —
 * and disable (rather than hide) when a capability is unavailable, with the
 * host's `disabledReason` explaining why.
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

// The verb row is wide at max-content; `contain: inline-size` zeroes its
// intrinsic width contribution so it can never stretch a host (queue row,
// preview footer) past its panel — it fills the given width and wraps.
const CONTAIN_INLINE_SIZE = { contain: 'inline-size' } as const;

export const RecallActionButtons = ({
  capabilities,
  children,
  disabledReason,
  onRecall,
}: {
  capabilities: ImageRecallCapabilities;
  /** Extra host-specific buttons rendered in the same group after the verbs. */
  children?: ReactNode;
  /** Tooltip for disabled verbs, explaining why recall is unavailable. */
  disabledReason?: string;
  onRecall: (kind: ImageRecallKind) => void;
}) => (
  <ButtonGroup css={CONTAIN_INLINE_SIZE} flexWrap="wrap" minW="0" rowGap="1" size="2xs" variant="subtle" w="full">
    {RECALL_ACTION_ITEMS.map((item) => (
      <RecallActionButton
        key={item.kind}
        disabledReason={disabledReason}
        icon={item.icon}
        isEnabled={capabilities[item.capability]}
        kind={item.kind}
        label={item.label}
        onRecall={onRecall}
      />
    ))}
    {children}
  </ButtonGroup>
);

const RecallActionButton = ({
  disabledReason,
  icon,
  isEnabled,
  kind,
  label,
  onRecall,
}: {
  disabledReason?: string;
  icon: LucideIcon;
  isEnabled: boolean;
  kind: ImageRecallKind;
  label: string;
  onRecall: (kind: ImageRecallKind) => void;
}) => {
  const handleClick = useCallback(() => onRecall(kind), [kind, onRecall]);
  const button = (
    <Button disabled={!isEnabled} onClick={handleClick}>
      <Icon as={icon} boxSize="3" />
      {label}
    </Button>
  );

  if (!isEnabled && disabledReason) {
    return <Tooltip content={disabledReason}>{button}</Tooltip>;
  }

  return button;
};
