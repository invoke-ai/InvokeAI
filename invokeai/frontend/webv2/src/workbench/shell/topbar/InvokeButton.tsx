import type { InvocationSourceId } from '@workbench/invocationContracts';
import type { FocusEvent, KeyboardEvent, MouseEvent } from 'react';

import {
  Box,
  HStack,
  Icon,
  Kbd,
  Popover,
  Portal,
  ProgressCircle,
  SegmentGroup,
  Separator,
  Spinner,
  Stack,
  Text,
} from '@chakra-ui/react';
import { getDeterminateProgressFraction } from '@features/queue/contracts';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button, useUnavailableControl } from '@platform/ui/Button';
import { PopoverContent } from '@platform/ui/Popover';
import { WidgetIcon } from '@workbench/iconResolver';
import { ROUTE_LABEL_KEYS } from '@workbench/invocation';
import { useActiveQueueProgress } from '@workbench/queue-integration/useActiveQueueProgress';
import { getWidgetById } from '@workbench/widgetRegistry';
import { useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { ArrowRightIcon, CircleAlertIcon, PlayIcon, WifiOffIcon } from 'lucide-react';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { InvocationIssue, InvocationState } from './useInvocationState';

import { getInvokeIconMode } from './invokeButtonModel';
import { useInvokeIssueActions } from './invokeIssueActions';
import { HIDE_BELOW_HINT_WIDTH } from './topbarBreakpoints';
import { TopbarShortcutKeys } from './TopbarShortcutKeys';
import { useTopbarShortcutBinding } from './useTopbarShortcut';

/**
 * Invoke sits on top of its attached neighbours with full control corners; the iterations field and routing button
 * each run a tail of their frame under those corners so they read as a continuing border, not a notch.
 * Doubled specificity outranks the attached Group's `!important` corner reset.
 */
const INVOKE_BUTTON_CSS = { '&&': { borderRadius: 'control !important' } } as const;
/** The hint inherits the button's text color; its frame is that color, faded. */
const SHORTCUT_BORDER = 'color-mix(in oklab, currentColor 40%, transparent)';
const POPOVER_POSITIONING = { placement: 'bottom-end', gutter: 6 } as const;
/** Frames a click waits for the list to mount before moving focus into it. */
const FOCUS_WAIT_FRAMES = 10;
/** Hover previews the issues after a beat, and forgives a short trip from the button to the card. */
const PREVIEW_OPEN_DELAY_MS = 250;
const PREVIEW_CLOSE_DELAY_MS = 200;

/**
 * `preview` follows hover and keyboard focus and leaves focus where it is; `pinned` is the blocked button's click, and
 * takes focus into the list so its fixes are a Tab away.
 */
type IssuesPopoverMode = 'closed' | 'pinned' | 'preview';

/**
 * Keep Invoke geometry and availability stable while work runs. Only the icon changes to progress; hover or
 * keyboard focus-visible restores play to signal that more work can queue. While issues block it, the button turns
 * secondary and opens the list of them, each with the way to its fix.
 */
export const InvokeButton = ({ state }: { state: InvocationState }) => {
  const { t } = useTranslation();
  const { invoke, isPreparing, isValid, issues } = state;
  const shortcutBinding = useTopbarShortcutBinding('app.invoke');
  const frontBinding = useTopbarShortcutBinding('app.invokeFront');
  const shortcutParts = shortcutBinding?.parts ?? null;
  const ariaKeyShortcuts =
    [shortcutBinding?.aria, frontBinding?.aria].filter((value) => value !== undefined).join(' ') || undefined;
  const contentId = useId();
  const buttonRef = useRef<HTMLButtonElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const timers = useRef<{ close?: number; open?: number }>({});
  const [mode, setMode] = useState<IssuesPopoverMode>('closed');
  // A jump to an issue's fix moves focus there; closing then must not pull it back to the button.
  const [returnsFocus, setReturnsFocus] = useState(true);

  const { progress: runningProgress, summary } = useActiveQueueProgress();
  const hasOpenWork = summary.total > 0;

  const [isHovered, setIsHovered] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  const canInvoke = isValid && !isPreparing;
  // Blocked by issues: the count replaces the shortcut, and activation opens the issue list.
  const showsIssues = !isValid && !isPreparing && issues.length > 0;

  const clearTimers = useCallback(() => {
    window.clearTimeout(timers.current.open);
    window.clearTimeout(timers.current.close);
    timers.current = {};
  }, []);
  const schedulePreview = useCallback(() => {
    window.clearTimeout(timers.current.close);
    timers.current.open = window.setTimeout(
      () => setMode((current) => (current === 'closed' ? 'preview' : current)),
      PREVIEW_OPEN_DELAY_MS
    );
  }, []);
  const schedulePreviewClose = useCallback(() => {
    window.clearTimeout(timers.current.open);
    timers.current.close = window.setTimeout(
      () => setMode((current) => (current === 'preview' ? 'closed' : current)),
      PREVIEW_CLOSE_DELAY_MS
    );
  }, []);

  // A preview timer must not outlive the button it would open the list for.
  useMountEffect(() => clearTimers);
  const handlePointerEnter = useCallback(() => {
    setIsHovered(true);
    schedulePreview();
  }, [schedulePreview]);
  const handlePointerLeave = useCallback(() => {
    setIsHovered(false);
    schedulePreviewClose();
  }, [schedulePreviewClose]);
  // Check focus-visible so mouse click-focus cannot pin the play icon throughout a batch.
  const handleFocus = useCallback(
    (event: FocusEvent<HTMLButtonElement>) => {
      if (event.currentTarget.matches(':focus-visible')) {
        setIsFocused(true);
        schedulePreview();
      }
    },
    [schedulePreview]
  );
  const handleBlur = useCallback(
    (event: FocusEvent<HTMLButtonElement>) => {
      setIsFocused(false);
      // Focus moving into the card keeps it; anywhere else ends a preview.
      if (!contentRef.current?.contains(event.relatedTarget as Node | null)) {
        schedulePreviewClose();
      }
    },
    [schedulePreviewClose]
  );
  // Focus goes to the list's first control once the list has mounted; opening on hover never moves it.
  const focusFirstControl = useCallback(() => {
    let frames = 0;
    const attempt = () => {
      // The first way out of the first issue, else the selected source: never a source the click did not choose.
      const control =
        contentRef.current?.querySelector<HTMLElement>('[data-issue-action]') ??
        contentRef.current?.querySelector<HTMLElement>('input[type="radio"]:checked');

      if (control) {
        control.focus();
      } else if (++frames < FOCUS_WAIT_FRAMES) {
        requestAnimationFrame(attempt);
      }
    };

    requestAnimationFrame(attempt);
  }, []);
  const handleClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      clearTimers();
      setReturnsFocus(true);

      if (canInvoke) {
        setMode('closed');
        // Shift+click is the pointer path to the front of the queue, beside the app.invokeFront shortcut.
        void invoke(event.shiftKey ? { prepend: true } : undefined);
        return;
      }

      if (!showsIssues) {
        return;
      }

      if (mode === 'pinned') {
        setMode('closed');
        return;
      }

      // Shown as a preview or not yet: either way the click pins it and moves focus to its first control.
      setMode('pinned');
      focusFirstControl();
    },
    [canInvoke, clearTimers, focusFirstControl, invoke, mode, showsIssues]
  );
  const handleOpenChange = useCallback(
    ({ open }: { open: boolean }) => {
      if (!open) {
        clearTimers();
        setReturnsFocus(true);
        setMode('closed');
      }
    },
    [clearTimers]
  );
  // The list sits in a portal: Tab past its last control (or Shift+Tab before its first) closes it and returns to
  // Invoke, rather than leaving focus on the page.
  const handleContentKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.key !== 'Tab') {
        return;
      }

      const controls = [
        ...event.currentTarget.querySelectorAll<HTMLElement>('button:not(:disabled), input:checked, [tabindex="0"]'),
      ].filter((control) => control.getClientRects().length > 0);
      const edge = event.shiftKey ? controls[0] : controls.at(-1);

      if (edge && document.activeElement === edge) {
        event.preventDefault();
        clearTimers();
        setReturnsFocus(true);
        setMode('closed');
      }
    },
    [clearTimers]
  );
  const handleJump = useCallback(() => {
    clearTimers();
    setReturnsFocus(false);
    setMode('closed');
  }, [clearTimers]);
  const getButton = useCallback(() => buttonRef.current, []);
  // The button is the popover's anchor, not its trigger: pressing it while open is its own toggle, not an outside press.
  const persistentElements = useMemo(() => [getButton], [getButton]);
  const popoverIds = useMemo(() => ({ content: contentId }), [contentId]);

  // Preparing is the one state with nothing to do: the reason describes the unavailable control.
  const unavailable = useUnavailableControl(isPreparing ? t('topbar.invoke.preparing') : undefined, undefined);

  const iconMode = isPreparing
    ? ({ mode: 'progress', value: null } as const)
    : getInvokeIconMode({
        hasOpenWork,
        isHovered: isHovered || isFocused,
        progress: getDeterminateProgressFraction(runningProgress?.percentage),
      });

  return (
    <Popover.Root
      autoFocus={false}
      finalFocusEl={getButton}
      ids={popoverIds}
      lazyMount
      open={mode !== 'closed'}
      persistentElements={persistentElements}
      positioning={POPOVER_POSITIONING}
      restoreFocus={returnsFocus}
      unmountOnExit
      onOpenChange={handleOpenChange}
    >
      <Popover.Anchor asChild>
        <Button
          {...unavailable?.props}
          ref={buttonRef}
          aria-controls={mode === 'closed' ? undefined : contentId}
          aria-expanded={showsIssues ? mode === 'pinned' : undefined}
          aria-haspopup={showsIssues ? 'dialog' : undefined}
          aria-keyshortcuts={ariaKeyShortcuts}
          colorPalette={showsIssues ? 'gray' : 'brand'}
          css={INVOKE_BUTTON_CSS}
          cursor={isPreparing ? 'not-allowed' : undefined}
          data-blocked={showsIssues ? '' : undefined}
          flexShrink={0}
          size="lg"
          // Blocked is not disabled: the button opens what is in the way. It steps back from the primary color so it
          // no longer reads as ready.
          variant={showsIssues ? 'outline' : 'solid'}
          onBlur={handleBlur}
          onClick={handleClick}
          onFocus={handleFocus}
          onPointerEnter={handlePointerEnter}
          onPointerLeave={handlePointerLeave}
          zIndex="2"
        >
          {/* Decorative: the queue's progress has its own status surfaces, and it must not change the button's name. */}
          <Box aria-hidden alignItems="center" boxSize="3.5" display="flex" justifyContent="center" position="relative">
            {showsIssues ? (
              <Icon as={CircleAlertIcon} boxSize="3.5" color="fg.error" />
            ) : iconMode.mode === 'play' ? (
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
          {unavailable?.description}
          {shortcutParts && !showsIssues ? (
            // The keycap stays out of the name; aria-keyshortcuts announces the binding.
            <Kbd
              aria-hidden
              css={HIDE_BELOW_HINT_WIDTH}
              variant="outline"
              borderColor={SHORTCUT_BORDER}
              color="inherit"
            >
              <TopbarShortcutKeys parts={shortcutParts} />
            </Kbd>
          ) : null}
        </Button>
      </Popover.Anchor>
      <Portal>
        <Popover.Positioner>
          <PopoverContent
            ref={contentRef}
            p="3"
            onKeyDown={handleContentKeyDown}
            onPointerEnter={clearTimers}
            onPointerLeave={mode === 'preview' ? schedulePreviewClose : undefined}
          >
            <InvokePopoverBody
              frontShortcut={frontBinding?.display ?? null}
              shortcutParts={shortcutParts}
              state={state}
              onJump={handleJump}
            />
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};

const InvokePopoverBody = ({
  frontShortcut,
  onJump,
  shortcutParts,
  state,
}: {
  frontShortcut: string | null;
  /** An action that leads elsewhere closes the list, leaving focus where the action put it. */
  onJump: () => void;
  shortcutParts: string[] | null;
  state: InvocationState;
}) => {
  const { t } = useTranslation();
  const { batchCount, invocation, isPreparing, isValid, issues, promptExpansion, workflowBatchSize } = state;
  const source = t(ROUTE_LABEL_KEYS[invocation.sourceId]);
  const destination = t(ROUTE_LABEL_KEYS[invocation.destination]);
  const promptCount = promptExpansion.count;
  const runs = t('topbar.invoke.summary.runs', { count: batchCount });
  const summary =
    invocation.sourceId === 'generate' || invocation.sourceId === 'upscale' || invocation.sourceId === 'video'
      ? promptExpansion.isLoading
        ? t('topbar.invoke.expandingPrompts')
        : t('topbar.invoke.summary.prompts', {
            generations: t('topbar.invoke.summary.generations', { count: promptCount * batchCount }),
            iterations: t('topbar.invoke.summary.iterations', { count: batchCount }),
            prompts: t('topbar.invoke.summary.promptCount', { count: promptCount }),
          })
      : workflowBatchSize === undefined
        ? t('topbar.invoke.summary.runsOnly', {
            generations: t('topbar.invoke.summary.generations', { count: batchCount }),
            runs,
            source,
          })
        : workflowBatchSize === null
          ? t('topbar.invoke.summary.batchPending', { runs, source })
          : t('topbar.invoke.summary.batch', {
              generations: t('topbar.invoke.summary.generations', { count: batchCount * workflowBatchSize }),
              items: t('topbar.invoke.summary.batchItems', { count: workflowBatchSize }),
              runs,
              source,
            });

  return (
    <Stack gap="2" maxW="22rem" minW="17rem">
      <HStack justify="space-between">
        <Popover.Title fontSize="sm" fontWeight="700">
          {isPreparing
            ? t('topbar.invoke.preparing')
            : isValid
              ? t('topbar.invoke.addToQueue')
              : t('topbar.invoke.unableToQueue')}
        </Popover.Title>
        {shortcutParts ? (
          <Kbd variant="subtle">
            <TopbarShortcutKeys parts={shortcutParts} />
          </Kbd>
        ) : null}
      </HStack>
      <InvokeRoute destination={destination} state={state} />
      <Text color="fg.muted" fontSize="xs">
        {summary}
      </Text>
      <Separator borderColor="border.subtle" />
      {isPreparing ? (
        <Text color="fg.muted" fontSize="sm">
          {t('topbar.invoke.preparing')}
        </Text>
      ) : issues.length > 0 ? (
        <InvokeIssueList issues={issues} onJump={onJump} />
      ) : (
        <Text color="fg.muted" fontSize="sm">
          {t('topbar.invoke.addingImagesTo', { destination })}
        </Text>
      )}
      <Text color="fg.subtle" fontSize="xs">
        {frontShortcut === null
          ? t('topbar.invoke.invokeFrontHintNoShortcut')
          : t('topbar.invoke.invokeFrontHint', { shortcut: frontShortcut })}
      </Text>
    </Stack>
  );
};

/**
 * Names what Invoke runs and where results go, and switches the source in place: among the sources this layout holds,
 * plus the current one when it is missing, so the choice is never between things the user cannot see.
 */
const InvokeRoute = ({ destination, state }: { destination: string; state: InvocationState }) => {
  const { t } = useTranslation();
  const { generation } = useWorkbenchCommands();
  const { invocation, placedTypeIds, sources } = state;
  const options = sources.filter(
    (candidate) => placedTypeIds.has(candidate.typeId) || candidate.sourceId === invocation.sourceId
  );
  const handleChange = useCallback(
    ({ value }: { value: string | null }) => {
      if (value) {
        generation.setSource(value as InvocationSourceId);
      }
    },
    [generation]
  );

  return (
    <Stack gap="1">
      <Text color="fg.subtle" fontSize="xs" fontWeight="700" textTransform="uppercase">
        {t('topbar.invoke.runs')}
      </Text>
      <HStack gap="2" wrap="wrap">
        {options.length > 1 ? (
          <SegmentGroup.Root
            aria-label={t('topbar.routing.source')}
            size="xs"
            value={invocation.sourceId}
            onValueChange={handleChange}
          >
            <SegmentGroup.Indicator />
            {options.map((option) => (
              <SegmentGroup.Item key={option.sourceId} value={option.sourceId}>
                <SegmentGroup.ItemHiddenInput />
                <SegmentGroup.ItemText alignItems="center" display="flex" fontSize="xs" gap="1">
                  <WidgetIcon boxSize="3" icon={getWidgetById(option.typeId)?.manifest.icon} />
                  {t(ROUTE_LABEL_KEYS[option.sourceId])}
                </SegmentGroup.ItemText>
              </SegmentGroup.Item>
            ))}
          </SegmentGroup.Root>
        ) : (
          <Text fontSize="sm" fontWeight="600">
            {t(ROUTE_LABEL_KEYS[invocation.sourceId])}
          </Text>
        )}
        <HStack color="fg.muted" gap="1">
          <Icon aria-hidden as={ArrowRightIcon} boxSize="3" />
          <Text fontSize="sm">{t('topbar.invoke.resultsTo', { destination })}</Text>
        </HStack>
      </HStack>
    </Stack>
  );
};

const InvokeIssueList = ({ issues, onJump }: { issues: readonly InvocationIssue[]; onJump: () => void }) => {
  const getActions = useInvokeIssueActions();

  return (
    <Stack as="ul" gap="2" listStyleType="none" m="0" p="0">
      {issues.map((issue, index) => {
        const actions = getActions(issue);

        return (
          // Two issues can read the same (two nodes missing the same input), so the position keeps keys apart.
          // oxlint-disable-next-line react/no-array-index-key -- the list is rebuilt whole on every change.
          <Stack key={`${index}:${issue.message}`} as="li" gap="1">
            <HStack align="start" gap="1.5">
              <Icon
                aria-hidden
                as={issue.target.kind === 'connection' ? WifiOffIcon : CircleAlertIcon}
                boxSize="3"
                color={issue.target.kind === 'connection' ? 'fg.muted' : 'fg.error'}
                flexShrink={0}
                mt="0.5"
              />
              <Text color="fg" fontSize="sm" lineHeight="1.4" textWrap="pretty">
                {issue.message}
              </Text>
            </HStack>
            {actions.length > 0 ? (
              <HStack gap="1" ps="4.5" wrap="wrap">
                {actions.map((action) => (
                  <Button
                    key={action.id}
                    size="xs"
                    data-issue-action=""
                    variant="outline"
                    // oxlint-disable-next-line react-perf/jsx-no-new-function-as-prop -- a few buttons per shown issue.
                    onClick={() => {
                      action.run();
                      if (!action.stays) {
                        onJump();
                      }
                    }}
                  >
                    {action.label}
                  </Button>
                ))}
              </HStack>
            ) : null}
          </Stack>
        );
      })}
    </Stack>
  );
};
