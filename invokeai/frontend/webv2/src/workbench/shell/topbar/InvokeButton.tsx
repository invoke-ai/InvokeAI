import type { FocusEvent, KeyboardEvent, MouseEvent } from 'react';

import { Box, Icon, Kbd, Popover, Portal, ProgressCircle, Spinner, Status } from '@chakra-ui/react';
import { getDeterminateProgressFraction } from '@features/queue/contracts';
import { useModifierHeld } from '@platform/react/useModifierHeld';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button, useUnavailableControl } from '@platform/ui/Button';
import { PopoverContent } from '@platform/ui/Popover';
import { useShortcutBinding } from '@workbench/hotkeys/useShortcutBinding';
import { isRoutingLocked } from '@workbench/invocation';
import { useActiveQueueProgress } from '@workbench/queue-integration/useActiveQueueProgress';
import { useNotify } from '@workbench/useNotify';
import { CircleAlertIcon, ListStartIcon, PlayIcon } from 'lucide-react';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type * as InvokePopoverBodyModule from './InvokePopoverBody';
import type { InvocationState } from './useInvocationState';

import { getInvokeIconMode } from './invokeButtonModel';
import { HIDE_BELOW_HINT_WIDTH } from './topbarBreakpoints';
import { TopbarShortcutKeys } from './TopbarShortcutKeys';

type InvokePopoverBodyComponent = typeof InvokePopoverBodyModule.InvokePopoverBody;

// The card and its issue fixes (which reach Generate, workflow and model owners) stay off editor boot. Hover and focus
// start the load; opening waits for it, so the card never shows empty.
let loadedPopoverBody: InvokePopoverBodyComponent | null = null;
let popoverBodyModule: Promise<InvokePopoverBodyComponent> | null = null;

/** A failed load is forgotten, so the next hover or click tries again. */
const loadPopoverBody = (): Promise<InvokePopoverBodyComponent> =>
  (popoverBodyModule ??= import('./InvokePopoverBody').then(
    (module) => (loadedPopoverBody = module.InvokePopoverBody),
    (error: unknown) => {
      popoverBodyModule = null;
      throw error;
    }
  ));

/**
 * Invoke ends its attached group and sits on top of the iterations field with full control corners; the field runs a
 * tail of its frame under Invoke's leading corners so it reads as a continuing border, not a notch.
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
 * `preview` follows hover and keyboard focus and leaves focus where it is; `pinned` stays until dismissed. The blocked
 * button's click, ArrowDown and a right-click pin it; any press inside a preview pins it too, so an edit there
 * survives a short pointer excursion.
 */
type IssuesPopoverMode = 'closed' | 'pinned' | 'preview';
/** An open waiting on the card's chunk. Compared by identity: a later request, even of the same mode, replaces it. */
interface PendingOpen {
  mode: Exclude<IssuesPopoverMode, 'closed'>;
}

/**
 * Keep Invoke geometry and availability stable while work runs. Only the icon changes to progress; hover or
 * keyboard focus-visible restores play to signal that more work can queue. While issues block it, the button turns
 * secondary and opens the list of them, each with the way to its fix.
 */
export const InvokeButton = ({ state }: { state: InvocationState }) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const { invoke, isPreparing, isValid, issues } = state;
  const shortcutBinding = useShortcutBinding('app.invoke');
  const frontBinding = useShortcutBinding('app.invokeFront');
  const shortcutParts = shortcutBinding?.parts ?? null;
  // Shift+click queues at the front: holding it over a ready button says so before the click.
  const isShiftHeld = useModifierHeld('Shift');
  const ariaKeyShortcuts =
    [shortcutBinding?.aria, frontBinding?.aria].filter((value) => value !== undefined).join(' ') || undefined;
  const contentId = useId();
  const buttonRef = useRef<HTMLButtonElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const timers = useRef<{ close?: number; open?: number }>({});
  const [mode, setMode] = useState<IssuesPopoverMode>('closed');
  // State, not the module variable, so the render that opens the card sees the body.
  const [PopoverBody, setPopoverBody] = useState<InvokePopoverBodyComponent | null>(() => loadedPopoverBody);
  // The open waiting on the body's chunk; any later open or close supersedes it. The ref answers the load's callbacks;
  // the state marks the button busy.
  const pendingOpen = useRef<PendingOpen | null>(null);
  const [isOpenPending, setIsOpenPending] = useState(false);

  const { progress: runningProgress, summary } = useActiveQueueProgress();
  const hasOpenWork = summary.total > 0;

  const [isHovered, setIsHovered] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  const canInvoke = isValid && !isPreparing;
  const isLocked = isRoutingLocked(state.invocation);
  // Blocked by issues: the count replaces the shortcut, and activation opens the issue list.
  const showsIssues = !isValid && !isPreparing && issues.length > 0;

  const setPendingOpen = useCallback((pending: PendingOpen | null) => {
    pendingOpen.current = pending;
    setIsOpenPending(pending !== null);
  }, []);
  const cancelPendingPin = useCallback(() => {
    if (pendingOpen.current?.mode === 'pinned') {
      setPendingOpen(null);
    }
  }, [setPendingOpen]);
  const clearTimers = useCallback(() => {
    window.clearTimeout(timers.current.open);
    window.clearTimeout(timers.current.close);
    timers.current = {};
    setPendingOpen(null);
  }, [setPendingOpen]);
  const openWhenLoaded = useCallback(
    (next: PendingOpen['mode'], onOpen?: () => void) => {
      // A click outranks hover: a pointer re-entering while the click's card loads must not demote it to a preview.
      if (next === 'preview' && pendingOpen.current?.mode === 'pinned') {
        return;
      }

      const open = (component: InvokePopoverBodyComponent) => {
        setPopoverBody(() => component);
        setMode((current) => (next === 'preview' && current !== 'closed' ? current : next));
        onOpen?.();
      };

      if (loadedPopoverBody) {
        setPendingOpen(null);
        open(loadedPopoverBody);
        return;
      }

      const request: PendingOpen = { mode: next };
      setPendingOpen(request);
      void loadPopoverBody().then(
        (component) => {
          if (pendingOpen.current === request) {
            setPendingOpen(null);
            open(component);
          }
        },
        () => {
          if (pendingOpen.current !== request) {
            return;
          }

          setPendingOpen(null);
          // A hover that came to nothing says nothing; a click gets an answer. The next attempt retries the load.
          if (next === 'pinned') {
            notify.error(t('topbar.invoke.cardLoadFailed'));
          }
        }
      );
    },
    [notify, setPendingOpen, t]
  );
  const schedulePreview = useCallback(() => {
    window.clearTimeout(timers.current.close);
    // Start the card's chunk now, so the preview after the delay (or a click before it) opens with its content.
    void loadPopoverBody().catch(() => {});
    timers.current.open = window.setTimeout(() => openWhenLoaded('preview'), PREVIEW_OPEN_DELAY_MS);
  }, [openWhenLoaded]);
  const schedulePreviewClose = useCallback(() => {
    window.clearTimeout(timers.current.open);
    if (pendingOpen.current?.mode === 'preview') {
      setPendingOpen(null);
    }
    timers.current.close = window.setTimeout(
      () => setMode((current) => (current === 'preview' ? 'closed' : current)),
      PREVIEW_CLOSE_DELAY_MS
    );
  }, [setPendingOpen]);

  // A preview timer must not outlive the button it would open the list for.
  useMountEffect(() => clearTimers);
  // A press elsewhere while a click's card loads takes the click back, even where pressing does not move focus, so a
  // late chunk never opens the card over what the person turned to.
  useMountEffect(() => {
    const handlePointerDown = (event: PointerEvent) => {
      if (!buttonRef.current?.contains(event.target as Node | null)) {
        cancelPendingPin();
      }
    };

    document.addEventListener('pointerdown', handlePointerDown, true);
    return () => document.removeEventListener('pointerdown', handlePointerDown, true);
  });
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
      // Focus moving into the card keeps it; anywhere else ends a preview, and a click's card still loading.
      if (!contentRef.current?.contains(event.relatedTarget as Node | null)) {
        cancelPendingPin();
        schedulePreviewClose();
      }
    },
    [cancelPendingPin, schedulePreviewClose]
  );
  // Focus goes to the list's first control once the list has mounted; opening on hover never moves it.
  const focusFirstControl = useCallback(() => {
    let frames = 0;
    const attempt = () => {
      // The first way out of the first issue, else the route's first selected segment (never one the user did not
      // choose), else its lock.
      const content = contentRef.current;
      const control =
        content?.querySelector<HTMLElement>('[data-issue-action]') ??
        content?.querySelector<HTMLElement>('[data-invoke-route] input[type="radio"]:checked') ??
        content?.querySelector<HTMLElement>('[data-routing-lock]');

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
      openWhenLoaded('pinned', focusFirstControl);
    },
    [canInvoke, clearTimers, focusFirstControl, invoke, mode, openWhenLoaded, showsIssues]
  );
  // Activation invokes while the route is valid, so the card has its own ways in: ArrowDown takes focus to its first
  // control, the way a menu button opens.
  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLButtonElement>) => {
      if (event.key !== 'ArrowDown' || event.ctrlKey || event.metaKey || event.shiftKey) {
        return;
      }

      // Handled here: the window's hotkeys (a layer nudge, gallery selection, prompt history) must not see it too.
      event.preventDefault();
      event.stopPropagation();
      clearTimers();
      openWhenLoaded('pinned', focusFirstControl);
    },
    [clearTimers, focusFirstControl, openWhenLoaded]
  );
  // The pointer's way in when a click would invoke; focus stays with the pointer.
  const handleContextMenu = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      clearTimers();
      openWhenLoaded('pinned');
    },
    [clearTimers, openWhenLoaded]
  );
  const pinPreview = useCallback(() => setMode((current) => (current === 'preview' ? 'pinned' : current)), []);
  // Focus comes back to Invoke only from inside the card. A preview closing, or an Escape or press while focus is
  // elsewhere, leaves it where it is: focus landing on Invoke unasked would make the next Enter queue.
  const closeCard = useCallback(() => {
    const returnsFocus = contentRef.current?.contains(document.activeElement) ?? false;

    clearTimers();
    setMode('closed');
    if (returnsFocus) {
      buttonRef.current?.focus();
    }
  }, [clearTimers]);
  const handleOpenChange = useCallback(
    ({ open }: { open: boolean }) => {
      if (!open) {
        closeCard();
      }
    },
    [closeCard]
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
        closeCard();
      }
    },
    [closeCard]
  );
  // A jump to an issue's fix moves focus there; closing must not pull it back to the button.
  const handleJump = useCallback(() => {
    clearTimers();
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
        isFrontHeld: isShiftHeld && canInvoke,
        isHovered: isHovered || isFocused,
        progress: getDeterminateProgressFraction(runningProgress?.percentage),
      });

  return (
    <Popover.Root
      autoFocus={false}
      ids={popoverIds}
      lazyMount
      open={mode !== 'closed'}
      persistentElements={persistentElements}
      positioning={POPOVER_POSITIONING}
      restoreFocus={false}
      unmountOnExit
      onOpenChange={handleOpenChange}
    >
      <Popover.Anchor asChild>
        <Button
          {...unavailable?.props}
          ref={buttonRef}
          aria-busy={isOpenPending || undefined}
          aria-controls={mode === 'closed' ? undefined : contentId}
          aria-expanded={mode === 'pinned'}
          aria-haspopup="dialog"
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
          onContextMenu={handleContextMenu}
          onFocus={handleFocus}
          onKeyDown={handleKeyDown}
          onPointerEnter={handlePointerEnter}
          onPointerLeave={handlePointerLeave}
          zIndex="2"
        >
          {/* Decorative: the queue's progress has its own status surfaces, and it must not change the button's name. */}
          <Box aria-hidden alignItems="center" boxSize="3.5" display="flex" justifyContent="center" position="relative">
            {isOpenPending ? (
              // The card is on its way: the slot says so until it opens.
              <Spinner borderWidth="1.5px" boxSize="3.5" color={showsIssues ? 'fg.muted' : 'bg'} />
            ) : showsIssues ? (
              <Icon as={CircleAlertIcon} boxSize="3.5" color="fg.error" />
            ) : iconMode.mode === 'play' ? (
              <Icon as={PlayIcon} boxSize="3.5" />
            ) : iconMode.mode === 'front' ? (
              <Icon as={ListStartIcon} boxSize="3.5" data-invoke-front-icon="" />
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
          {isLocked ? (
            // Shows the route is locked while the card is closed; the card's lock control names it.
            <Status.Root
              aria-hidden
              colorPalette="accent"
              data-routing-lock-indicator=""
              pointerEvents="none"
              position="absolute"
              right="-1px"
              top="-1px"
            >
              <Status.Indicator />
            </Status.Root>
          ) : null}
        </Button>
      </Popover.Anchor>
      <Portal>
        <Popover.Positioner>
          <PopoverContent
            ref={contentRef}
            p="3"
            onKeyDown={handleContentKeyDown}
            onPointerDown={pinPreview}
            onPointerEnter={clearTimers}
            onPointerLeave={mode === 'preview' ? schedulePreviewClose : undefined}
          >
            {PopoverBody ? (
              <PopoverBody
                frontShortcut={frontBinding?.display ?? null}
                shortcutParts={shortcutParts}
                state={state}
                onJump={handleJump}
              />
            ) : null}
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};
