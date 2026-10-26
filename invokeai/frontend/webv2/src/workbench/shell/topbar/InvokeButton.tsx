import type { FocusEvent, MouseEvent } from 'react';

import { Box, HStack, Icon, Kbd, ProgressCircle, Separator, Spinner, Stack, Text } from '@chakra-ui/react';
import { getDeterminateProgressFraction } from '@features/queue/contracts';
import { Button } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { getDestinationLabel } from '@workbench/invocation';
import { useActiveQueueProgress } from '@workbench/queue-integration/useActiveQueueProgress';
import { useRevealLayerProperties } from '@workbench/widgets/layers/revealLayerProperties';
import { PlayIcon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { InvocationState } from './useInvocationState';

import { getInvokeIconMode } from './invokeButtonModel';
import { HIDE_BELOW_HINT_WIDTH } from './topbarBreakpoints';
import { TopbarShortcutKeys } from './TopbarShortcutKeys';
import { useTopbarShortcutBinding } from './useTopbarShortcut';

const TOOLTIP_CONTENT_PROPS = { p: '0' };

/**
 * Invoke sits on top of its attached neighbours with full control corners; the iterations field and routing button
 * each run a tail of their frame under those corners so they read as a continuing border, not a notch.
 * Doubled specificity outranks the attached Group's `!important` corner reset.
 */
const INVOKE_BUTTON_CSS = { '&&': { borderRadius: 'control !important' } } as const;
/** The hint inherits the button's text color; its frame is that color, faded. */
const SHORTCUT_BORDER = 'color-mix(in oklab, currentColor 40%, transparent)';

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
  const { invoke, isPreparing, isValid, issues } = state;
  const revealLayerProperties = useRevealLayerProperties();
  const shortcutBinding = useTopbarShortcutBinding('app.invoke');
  const frontBinding = useTopbarShortcutBinding('app.invokeFront');
  const shortcutParts = shortcutBinding?.parts ?? null;
  const frontShortcut = frontBinding?.display ?? null;
  const ariaKeyShortcuts =
    [shortcutBinding?.aria, frontBinding?.aria].filter((value) => value !== undefined).join(' ') || undefined;
  const tooltipContent = useMemo(
    () => <InvokeTooltipContent frontShortcut={frontShortcut} shortcutParts={shortcutParts} state={state} />,
    [frontShortcut, shortcutParts, state]
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
  // Shift+click is the pointer path to the front of the queue, beside the app.invokeFront shortcut.
  const handleClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => void invoke(event.shiftKey ? { prepend: true } : undefined),
    [invoke]
  );
  const canInvoke = isValid && !isPreparing;
  // Blocked by issues: the count replaces the shortcut, and when the first names a layer, activation shows it.
  const showsIssues = !isValid && !isPreparing && issues.length > 0;
  const issueLayerId = showsIssues ? (issues[0]?.layerId ?? null) : null;
  const handleShowIssue = useCallback(() => {
    if (issueLayerId !== null) {
      revealLayerProperties(issueLayerId);
    }
  }, [issueLayerId, revealLayerProperties]);

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
        aria-disabled={!canInvoke && issueLayerId === null}
        aria-keyshortcuts={ariaKeyShortcuts}
        aria-label={
          canInvoke
            ? t('topbar.invoke.invoke')
            : isPreparing
              ? t('topbar.invoke.preparing')
              : issueLayerId !== null
                ? t('topbar.invoke.showIssue', { count: issues.length, reason: issues[0]?.message })
                : t('topbar.invoke.unavailable', {
                    reason: issues[0]?.message ?? t('topbar.invoke.unrunnable'),
                  })
        }
        colorPalette="brand"
        css={INVOKE_BUTTON_CSS}
        cursor={canInvoke || issueLayerId !== null ? undefined : 'not-allowed'}
        flexShrink={0}
        // A blocked Invoke that can show its first issue is still actionable, so it is not dimmed.
        opacity={canInvoke || issueLayerId !== null ? undefined : 0.55}
        size="lg"
        onBlur={handleBlur}
        onClick={canInvoke ? handleClick : issueLayerId !== null ? handleShowIssue : undefined}
        onFocus={handleFocus}
        onPointerEnter={handlePointerEnter}
        onPointerLeave={handlePointerLeave}
        zIndex="2"
      >
        <Box alignItems="center" boxSize="3.5" display="flex" justifyContent="center" position="relative">
          {iconMode.mode === 'play' ? (
            <Icon as={PlayIcon} boxSize="3.5" />
          ) : iconMode.value === null ? (
            // The app-wide spinner, not the ring's indeterminate sweep, for unknown progress.
            <Spinner borderWidth="1.5px" boxSize="3.5" color="bg" />
          ) : (
            <ProgressCircle.Root size="xs" value={iconMode.value * 100}>
              <ProgressCircle.Circle>
                <ProgressCircle.Track stroke="bg/40" />
                <ProgressCircle.Range stroke="bg" strokeLinecap="round" />
              </ProgressCircle.Circle>
            </ProgressCircle.Root>
          )}
        </Box>
        {showsIssues ? t('topbar.invoke.issueCount', { count: issues.length }) : t('topbar.invoke.invoke')}
        {shortcutParts && !showsIssues ? (
          <Kbd css={HIDE_BELOW_HINT_WIDTH} variant="outline" borderColor={SHORTCUT_BORDER} color="inherit">
            <TopbarShortcutKeys parts={shortcutParts} />
          </Kbd>
        ) : null}
      </Button>
    </Tooltip>
  );
};

const InvokeTooltipContent = ({
  frontShortcut,
  shortcutParts,
  state,
}: {
  frontShortcut: string | null;
  shortcutParts: string[] | null;
  state: InvocationState;
}) => {
  const { t } = useTranslation();
  const { batchCount, invocation, isPreparing, isValid, issues, promptExpansion, workflowBatchSize } = state;
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
        <Text fontSize="md" fontWeight="800">
          {isPreparing
            ? t('topbar.invoke.preparing')
            : isValid
              ? t('topbar.invoke.addToQueue')
              : t('topbar.invoke.unableToQueue')}
        </Text>
        {shortcutParts ? (
          <Kbd variant="subtle">
            <TopbarShortcutKeys parts={shortcutParts} />
          </Kbd>
        ) : null}
      </HStack>
      <Text color="fg.muted" fontSize="md">
        {summary}
      </Text>
      <Separator borderColor="border.subtle" />
      {isPreparing ? (
        <Text color="fg.muted" fontSize="md">
          {t('topbar.invoke.preparing')}
        </Text>
      ) : issues.length > 0 ? (
        <Stack gap="1">
          {issues.map(({ message: reason }) => (
            <HStack key={reason} align="start" gap="1.5">
              <Text color="fg.subtle" fontSize="md" lineHeight="1.35">
                •
              </Text>
              <Text color="fg.muted" fontSize="md" lineHeight="1.35">
                {compactBlockingReason(reason, t('topbar.invoke.noNodes'))}
              </Text>
            </HStack>
          ))}
        </Stack>
      ) : (
        <Text color="fg.muted" fontSize="md">
          {t('topbar.invoke.addingImagesTo', { destination })}
        </Text>
      )}
      <Text color="fg.subtle" fontSize="sm">
        {frontShortcut === null
          ? t('topbar.invoke.invokeFrontHintNoShortcut')
          : t('topbar.invoke.invokeFrontHint', { shortcut: frontShortcut })}
      </Text>
    </Stack>
  );
};
