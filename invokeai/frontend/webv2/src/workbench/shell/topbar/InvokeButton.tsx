import type { ComponentProps } from 'react';

import { Box, HStack, Icon, Kbd, ProgressCircle, Separator, Stack, Text } from '@chakra-ui/react';
import { getQueueSummary } from '@features/queue/contracts';
import { useQueueItemProgress } from '@features/queue/react';
import { Button } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { getDestinationLabel } from '@workbench/invocation';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { PlayIcon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { InvocationState } from './useInvocationState';

import { getInvokeIconMode } from './invokeButtonModel';
import { HIDE_BELOW_HINT_WIDTH } from './topbarBreakpoints';
import { TopbarShortcutKeys } from './TopbarShortcutKeys';
import { useTopbarShortcutBinding } from './useTopbarShortcut';

const TOOLTIP_CONTENT_PROPS = { p: '0' };

type ProgressCircleRootProps = ComponentProps<typeof ProgressCircle.Root>;
// The `3xs` size (14px/2px, defined in platform/ui/theme/recipes.ts) is a repo
// extension the generated Chakra types don't know about yet; QueueProgressIndicator
// casts through the same seam for its `2xs` extension.
const ICON_RING_SIZE = '3xs' as ProgressCircleRootProps['size'];

const compactBlockingReason = (reason: string, noNodesLabel: string): string => {
  if (reason === 'The project graph has no nodes. Add nodes in the Workflow view.') {
    return noNodesLabel;
  }

  return reason.replace(/^The /, '').replace(/project graph/i, 'workflow');
};

const plural = (count: number, noun: string): string => `${count} ${noun}${count === 1 ? '' : 's'}`;

/**
 * The single Invoke action for the whole application.
 *
 * Its geometry, width, and enabled state never change while a batch runs —
 * Invoke queues further items on top of a running batch, so a button that
 * morphed into a progress bar would read as unavailable at exactly the
 * moment it is most useful. Only the icon slot's content may change: it
 * shows the queue's progress while a batch runs and the pointer is
 * elsewhere, and reverts to the play glyph on hover so "queue more on top"
 * always reads as available. Aggregate progress otherwise belongs to the
 * queue group. (§5.1, contract §9.4.)
 */
export const InvokeButton = ({ state }: { state: InvocationState }) => {
  const { t } = useTranslation();
  const { blockingReasons, invoke, isValid } = state;
  const shortcutBinding = useTopbarShortcutBinding('app.invoke');
  const shortcutParts = shortcutBinding?.parts ?? null;
  const tooltipContent = useMemo(
    () => <InvokeTooltipContent shortcutParts={shortcutParts} state={state} />,
    [shortcutParts, state]
  );

  const queueItems = useActiveProjectSelector((project) => project.queue.items);
  const baseSummary = getQueueSummary(queueItems);
  const runningProgress = useQueueItemProgress(baseSummary.runningQueueItemId ?? '');
  const hasOpenWork = baseSummary.total > 0;

  const [isHovered, setIsHovered] = useState(false);
  const handlePointerEnter = useCallback(() => setIsHovered(true), []);
  const handlePointerLeave = useCallback(() => setIsHovered(false), []);
  const handleClick = useCallback(() => void invoke(), [invoke]);

  const iconMode = getInvokeIconMode({
    hasOpenWork,
    isHovered,
    progress: runningProgress?.percentage ?? null,
  });

  return (
    <Tooltip content={tooltipContent} contentProps={TOOLTIP_CONTENT_PROPS} openDelay={200} showArrow>
      <Button
        aria-disabled={!isValid}
        aria-keyshortcuts={shortcutBinding?.aria}
        aria-label={
          isValid
            ? t('topbar.invoke.invoke')
            : t('topbar.invoke.unavailable', {
                reason: blockingReasons[0] ?? t('topbar.invoke.unrunnable'),
              })
        }
        colorPalette="brand"
        cursor={isValid ? undefined : 'not-allowed'}
        flexShrink={0}
        opacity={isValid ? undefined : 0.55}
        size="xs"
        onClick={isValid ? handleClick : undefined}
        onPointerEnter={handlePointerEnter}
        onPointerLeave={handlePointerLeave}
        zIndex="2"
      >
        <Box alignItems="center" boxSize="3.5" display="flex" justifyContent="center" position="relative">
          {iconMode.mode === 'progress' ? (
            <ProgressCircle.Root size={ICON_RING_SIZE} value={iconMode.value === null ? null : iconMode.value * 100}>
              <ProgressCircle.Circle>
                <ProgressCircle.Track stroke="bg/40" />
                <ProgressCircle.Range stroke="bg" strokeLinecap="round" />
              </ProgressCircle.Circle>
            </ProgressCircle.Root>
          ) : (
            <Icon as={PlayIcon} boxSize="3.5" />
          )}
        </Box>
        {t('topbar.invoke.invoke')}
        {shortcutParts ? (
          <Kbd css={HIDE_BELOW_HINT_WIDTH} variant="outline" color="bg" size="sm">
            <TopbarShortcutKeys parts={shortcutParts} />
          </Kbd>
        ) : null}
      </Button>
    </Tooltip>
  );
};

const InvokeTooltipContent = ({ shortcutParts, state }: { shortcutParts: string[] | null; state: InvocationState }) => {
  const { t } = useTranslation();
  const { batchCount, blockingReasons, invocation, isValid, promptExpansion } = state;
  const destination = getDestinationLabel(invocation.destination);
  const promptCount = promptExpansion.count;
  const summary =
    invocation.sourceId === 'generate' || invocation.sourceId === 'upscale'
      ? promptExpansion.isLoading
        ? t('topbar.invoke.expandingPrompts')
        : `${plural(promptCount, 'prompt')} × ${plural(batchCount, 'iteration')} → ${plural(promptCount * batchCount, 'generation')}`
      : `Workflow × ${plural(batchCount, 'run')} → ${plural(batchCount, 'generation')}`;

  return (
    <Stack gap="1.5" minW="14rem" p="2">
      <HStack justify="space-between">
        <Text fontSize="xs" fontWeight="800">
          {isValid ? t('topbar.invoke.addToQueue') : t('topbar.invoke.unableToQueue')}
        </Text>
        {shortcutParts ? (
          <Kbd size="sm" variant="subtle">
            <TopbarShortcutKeys parts={shortcutParts} />
          </Kbd>
        ) : null}
      </HStack>
      <Text color="fg.muted" fontSize="xs">
        {summary}
      </Text>
      <Separator borderColor="border.subtle" />
      {blockingReasons.length > 0 ? (
        <Stack gap="1">
          {blockingReasons.map((reason) => (
            <HStack key={reason} align="start" gap="1.5">
              <Text color="fg.subtle" fontSize="xs" lineHeight="1.35">
                •
              </Text>
              <Text color="fg.muted" fontSize="xs" lineHeight="1.35">
                {compactBlockingReason(reason, t('topbar.invoke.noNodes'))}
              </Text>
            </HStack>
          ))}
        </Stack>
      ) : (
        <Text color="fg.muted" fontSize="xs">
          {t('topbar.invoke.addingImagesTo', { destination })}
        </Text>
      )}
    </Stack>
  );
};
