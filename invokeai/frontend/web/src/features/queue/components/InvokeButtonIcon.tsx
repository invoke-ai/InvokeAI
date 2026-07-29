import { Icon, spinAnimation, useShiftModifier } from '@invoke-ai/ui-library';
import { useUserHasActiveQueueItems } from 'features/queue/hooks/useUserHasActiveQueueItems';
import { memo } from 'react';
import { PiCircleNotchBold, PiLightningFill, PiSparkleFill } from 'react-icons/pi';

type Props = {
  /** Whether the invoke button is disabled — the parent already subscribes to useInvoke, so it
   * passes the flag down rather than the icon duplicating that (heavy) hook tree. */
  isDisabled: boolean;
  boxSize: number;
};

/**
 * Icon shared by the invoke buttons: a spinner while the user has queue items pending or in
 * progress — feedback that their session is enqueued even when another user's generation
 * occupies the processor — otherwise the shift-dependent lightning/sparkle glyph.
 */
export const InvokeButtonIcon = memo(({ isDisabled, boxSize }: Props) => {
  const shift = useShiftModifier();
  const hasActiveQueueItems = useUserHasActiveQueueItems();

  if (!isDisabled && hasActiveQueueItems) {
    return <Icon boxSize={boxSize} as={PiCircleNotchBold} animation={spinAnimation} />;
  }

  if (shift) {
    return <PiLightningFill />;
  }

  return <PiSparkleFill />;
});
InvokeButtonIcon.displayName = 'InvokeButtonIcon';
