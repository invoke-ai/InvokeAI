import type { CanvasInteractionStateCapability, LayerAdjustmentStatus } from '@workbench/canvas-engine/api';
import type { TFunction } from 'i18next';

import { Box, HStack, Icon, Spinner, Text } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Tooltip } from '@platform/ui/Tooltip';
import { TriangleAlertIcon } from 'lucide-react';
import { useCallback, useState, useSyncExternalStore } from 'react';
import { useTranslation } from 'react-i18next';

/** The engine slice reporting whether an owner's spatial adjustments (Clarity, Texture, Dehaze) are shown exactly. */
export type AdjustmentStatusEngine = {
  readonly interaction: Pick<
    CanvasInteractionStateCapability,
    'getLayerAdjustmentStatus' | 'subscribeLayerAdjustmentStatus'
  >;
};

export type AdjustmentOwnerStatus = LayerAdjustmentStatus | 'idle';

/** A job shorter than this never shows the spinner, so quick updates while scrubbing do not flash. */
export const ADJUSTMENT_SPINNER_DELAY_MS = 250;

/** One owner's status, layer or group alike (the engine keys both by node id); idle without an engine. */
export const useAdjustmentStatus = (engine: AdjustmentStatusEngine | null, ownerId: string): AdjustmentOwnerStatus => {
  const subscribe = useCallback(
    (listener: () => void) =>
      engine?.interaction.subscribeLayerAdjustmentStatus(ownerId, listener) ?? (() => undefined),
    [engine, ownerId]
  );
  const getSnapshot = useCallback(
    () => engine?.interaction.getLayerAdjustmentStatus(ownerId) ?? 'idle',
    [engine, ownerId]
  );
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
};

/** Why the canvas leaves an owner's spatial adjustments out and what to do, or null while they apply. */
export const adjustmentWarningMessage = (t: TFunction, status: AdjustmentOwnerStatus): string | null => {
  switch (status) {
    case 'too-large':
      return t('widgets.layers.adjustments.status.tooLarge');
    case 'over-budget':
      return t('widgets.layers.adjustments.status.overBudget');
    case 'error':
      return t('widgets.layers.adjustments.status.error');
    case 'computing':
    case 'idle':
      return null;
  }
};

/** Mounted for one run of `computing`; it appears only once the job has outlasted the delay. */
const DelayedSpinner = () => {
  const { t } = useTranslation();
  const [revealed, setRevealed] = useState(false);
  useMountEffect(() => {
    const timer = window.setTimeout(() => setRevealed(true), ADJUSTMENT_SPINNER_DELAY_MS);
    return () => window.clearTimeout(timer);
  });
  if (!revealed) {
    return null;
  }
  const label = t('widgets.layers.adjustments.status.computing');
  return (
    <Tooltip content={label}>
      <Spinner aria-label={label} boxSize="3" color="fg.muted" flexShrink={0} role="img" size="inherit" />
    </Tooltip>
  );
};

/**
 * A layer or group row's spatial adjustment indicator: a delayed spinner while a job runs, a warning when the canvas
 * cannot apply them. `descriptionId` lets the row point its accessible description at the warning, since the row's
 * controls are pointer-only.
 */
export const AdjustmentStatusIcon = ({
  descriptionId,
  status,
}: {
  descriptionId: string;
  status: AdjustmentOwnerStatus;
}) => {
  const { t } = useTranslation();
  if (status === 'computing') {
    return <DelayedSpinner />;
  }
  const message = adjustmentWarningMessage(t, status);
  if (!message) {
    return null;
  }
  return (
    <Tooltip content={message}>
      <Icon
        aria-label={message}
        as={TriangleAlertIcon}
        boxSize="3"
        color="fg.warning"
        flexShrink={0}
        id={descriptionId}
        role="img"
      />
    </Tooltip>
  );
};

/**
 * The same warning, in words, above a spatial adjustment's settings, where the keyboard reaches it. The empty live
 * region stays mounted so screen readers announce a warning when it appears.
 */
export const AdjustmentStatusNotice = ({
  engine,
  ownerId,
}: {
  engine: AdjustmentStatusEngine | null;
  ownerId: string;
}) => {
  const { t } = useTranslation();
  const message = adjustmentWarningMessage(t, useAdjustmentStatus(engine, ownerId));
  // Spaced below by its own padding, since an empty region must not add a gap.
  return (
    <Box role="status">
      {message ? (
        <HStack align="start" gap="1.5" pb="3">
          <Icon as={TriangleAlertIcon} aria-hidden boxSize="3.5" color="fg.warning" flexShrink={0} mt="0.5" />
          <Text fontSize="md">{message}</Text>
        </HStack>
      ) : null}
    </Box>
  );
};
