import type { ComponentProps, FocusEvent } from 'react';

import { Box, HStack, Icon, Kbd, ProgressCircle, Separator, Stack, Text } from '@chakra-ui/react';
import { getDeterminateProgressFraction } from '@features/queue/contracts';
import { Button } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { getDestinationLabel } from '@workbench/invocation';
import { useActiveQueueProgress } from '@workbench/queue-integration/useActiveQueueProgress';
import { PlayIcon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { InvocationState } from './useInvocationState';

import { getInvokeIconMode } from './invokeButtonModel';
import { HIDE_BELOW_HINT_WIDTH } from './topbarBreakpoints';
import { TopbarShortcutKeys } from './TopbarShortcutKeys';
import { useTopbarShortcutBinding } from './useTopbarShortcut';

const TOOLTIP_CONTENT_PROPS = { p: '0' };
/** The hint inherits the button's text color; its frame is that color, faded. */
const SHORTCUT_BORDER = 'color-mix(in oklab, currentColor 40%, transparent)';

type ProgressCircleRootProps = ComponentProps<typeof ProgressCircle.Root>;
// Cast the repository's 3xs theme extension, which generated Chakra types do not yet include.
const ICON_RING_SIZE = '3xs' as ProgressCircleRootProps['size'];

const compactBlockingReason = (reason: string, noNodesLabel: string): string => {
  if (reason === 'The project graph has no nodes. Add nodes in the Workflow view.') {
    return noNodesLabel;
  }

  return reason.replace(/^The /, '').replace(/project graph/i, 'workflow');
};

const plural = (count: number, noun: string): string => `${count} ${noun}${count === 1 ? '' : 's'}`;

/**
 * Keep Invoke geometry and availability stable while work runs. Only the icon changes to progress; hover or
 * keyboard focus-visible restores play to signal that more work can queue.
 */
export const InvokeButton = ({ state }: { state: InvocationState }) => {
  const { t } = useTranslation();
  const { blockingReasons, invoke, isPreparing, isValid } = state;
  const shortcutBinding = useTopbarShortcutBinding('app.invoke');
  const shortcutParts = shortcutBinding?.parts ?? null;
  const tooltipContent = useMemo(
    () => <InvokeTooltipContent shortcutParts={shortcutParts} state={state} />,
    [shortcutParts, state]
  );

  const { progress: runningProgress, summary } = useActiveQueueProgress();
  const hasOpenWork = summary.total > 0;

  const [isHovered, setIsHovered] = useState(false);
  const handlePointerEnter = useCallback(() => setIsHovered(true), []);
  const handlePointerLeave = useCallback(() => setIsHovered(false), []);
  const [isFocused, setIsFocused] = useState(false);
  // Check focus-visible so mouse click-focus cannot pin the play icon throughout a batch.
  const handleFocus = useCallback((event: FocusEvent<HTMLButtonElement>) => {
    if (event.currentTarget.matches(':focus-visible')) {
      setIsFocused(true);
    }
  }, []);
  const handleBlur = useCallback(() => setIsFocused(false), []);
  const handleClick = useCallback(() => void invoke(), [invoke]);
  const canInvoke = isValid && !isPreparing;

  const iconMode = isPreparing
    ? ({ mode: 'progress', value: null } as const)
    : getInvokeIconMode({
        hasOpenWork,
        isHovered: isHovered || isFocused,
        progress: getDeterminateProgressFraction(runningProgress?.percentage),
      });

  return (
    <Tooltip content={tooltipContent} contentProps={TOOLTIP_CONTENT_PROPS} openDelay={200} showArrow>
      <Button
        aria-disabled={!canInvoke}
        aria-keyshortcuts={shortcutBinding?.aria}
        aria-label={
          canInvoke
            ? t('topbar.invoke.invoke')
            : isPreparing
              ? t('topbar.invoke.preparing')
              : t('topbar.invoke.unavailable', {
                  reason: blockingReasons[0] ?? t('topbar.invoke.unrunnable'),
                })
        }
        colorPalette="brand"
        cursor={canInvoke ? undefined : 'not-allowed'}
        flexShrink={0}
        opacity={canInvoke ? undefined : 0.55}
        size="sm"
        onBlur={handleBlur}
        onClick={canInvoke ? handleClick : undefined}
        onFocus={handleFocus}
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
          <Kbd css={HIDE_BELOW_HINT_WIDTH} variant="outline" borderColor={SHORTCUT_BORDER} color="inherit" size="sm">
            <TopbarShortcutKeys parts={shortcutParts} />
          </Kbd>
        ) : null}
      </Button>
    </Tooltip>
  );
};

const InvokeTooltipContent = ({ shortcutParts, state }: { shortcutParts: string[] | null; state: InvocationState }) => {
  const { t } = useTranslation();
  const { batchCount, blockingReasons, invocation, isPreparing, isValid, promptExpansion, workflowBatchSize } = state;
  const destination = getDestinationLabel(invocation.destination);
  const promptCount = promptExpansion.count;
  const summary =
    invocation.sourceId === 'generate' || invocation.sourceId === 'upscale' || invocation.sourceId === 'video'
      ? promptExpansion.isLoading
        ? t('topbar.invoke.expandingPrompts')
        : `${plural(promptCount, 'prompt')} × ${plural(batchCount, 'iteration')} → ${plural(promptCount * batchCount, 'generation')}`
      : workflowBatchSize === undefined
        ? `Workflow × ${plural(batchCount, 'run')} → ${plural(batchCount, 'generation')}`
        : workflowBatchSize === null
          ? `Workflow × ${plural(batchCount, 'run')} × batch (size resolves on invoke)`
          : `Workflow × ${plural(batchCount, 'run')} × ${plural(workflowBatchSize, 'batch item')} → ${plural(batchCount * workflowBatchSize, 'generation')}`;

  return (
    <Stack gap="1.5" minW="14rem" p="2">
      <HStack justify="space-between">
        <Text fontSize="xs" fontWeight="800">
          {isPreparing
            ? t('topbar.invoke.preparing')
            : isValid
              ? t('topbar.invoke.addToQueue')
              : t('topbar.invoke.unableToQueue')}
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
      {isPreparing ? (
        <Text color="fg.muted" fontSize="xs">
          {t('topbar.invoke.preparing')}
        </Text>
      ) : blockingReasons.length > 0 ? (
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
