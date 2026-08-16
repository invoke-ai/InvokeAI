import { HStack, Icon, Kbd, Separator, Stack, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { getDestinationLabel } from '@workbench/invocation';
import { PlayIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { InvocationState } from './useInvocationState';

import { HIDE_BELOW_HINT_WIDTH } from './topbarBreakpoints';
import { useTopbarShortcutBinding } from './useTopbarShortcut';

const TOOLTIP_CONTENT_PROPS = { p: '0' };

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
 * Its appearance, width, and enabled state never change while a batch runs.
 * Invoke queues further items on top of a running batch, so a button that
 * morphed into a progress bar would read as unavailable at exactly the moment
 * it is most useful. Progress belongs to the queue group. (§5.1, contract §9.4.)
 */
export const InvokeButton = ({ state }: { state: InvocationState }) => {
  const { t } = useTranslation();
  const { blockingReasons, invoke, isValid } = state;
  const shortcutBinding = useTopbarShortcutBinding('app.invoke');
  const shortcut = shortcutBinding?.display ?? null;
  const handleClick = useCallback(() => void invoke(), [invoke]);
  const tooltipContent = useMemo(() => <InvokeTooltipContent shortcut={shortcut} state={state} />, [shortcut, state]);

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
        zIndex="2"
      >
        <Icon as={PlayIcon} />
        {t('topbar.invoke.invoke')}
        {shortcut ? (
          <Kbd css={HIDE_BELOW_HINT_WIDTH} variant="outline" color="bg" size="sm">
            {shortcut}
          </Kbd>
        ) : null}
      </Button>
    </Tooltip>
  );
};

const InvokeTooltipContent = ({ shortcut, state }: { shortcut: string | null; state: InvocationState }) => {
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
        {shortcut ? (
          <Kbd size="sm" variant="subtle">
            {shortcut}
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
