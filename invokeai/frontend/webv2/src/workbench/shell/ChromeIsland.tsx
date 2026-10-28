import type { MouseEvent, ReactNode } from 'react';

import { Flex, HStack, Icon } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { forwardWheelToStrip, wheelScrollsHorizontally } from '@platform/ui/wheelScrollsHorizontally';
import { ChevronLeftIcon, ChevronRightIcon } from 'lucide-react';
import { useCallback, useState, useSyncExternalStore } from 'react';
import { useTranslation } from 'react-i18next';

type Edge = 'end' | 'start';

interface HiddenEdges {
  end: boolean;
  start: boolean;
}

const NO_HIDDEN_EDGES: HiddenEdges = { end: false, start: false };

// A fractional zoom can leave a sub-pixel remainder at either end of a fully scrolled row; that hides nothing.
const EDGE_TOLERANCE_PX = 1;

/** Fade plus the cue's button and the island's inline padding beside it. */
const CUE_WIDTH = '2.75rem';
/** The row's inline padding; the cue covers the end padding, the start padding stays in front of the first control. */
const ROW_PADDING_X = '1';
const CUE_FADE = '1rem';

const fadeToward = (side: 'left' | 'right') =>
  `linear-gradient(to ${side}, transparent, {colors.bg.subtle} ${CUE_FADE})`;

// The fade runs from the row into the island's own background at the edge hiding content; right-to-left mirrors it.
const CUE_CSS: Record<Edge, object> = {
  end: { bgImage: fadeToward('right'), _rtl: { bgImage: fadeToward('left') } },
  start: { bgImage: fadeToward('left'), _rtl: { bgImage: fadeToward('right') } },
};

const MIRRORED_IN_RTL = { _rtl: { transform: 'scaleX(-1)' } } as const;

const HIDDEN_SCROLLBAR_CSS = { '&::-webkit-scrollbar': { display: 'none' }, scrollbarWidth: 'none' } as const;

const prefersReducedMotion = (): boolean =>
  document.documentElement.dataset.reduceMotion === 'true' ||
  globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;

/**
 * Keep focus on the control the user was on, whether the press lands on the button or the fade: the cue is a pointer
 * shortcut, and focus already scrolls the row.
 */
const keepFocus = (event: MouseEvent) => event.preventDefault();

/**
 * The narrowest island that still shows `controlWidth` of its row's start clear of the end cue: border, start padding,
 * the control, and the cue.
 */
export const getChromeIslandMinWidth = (controlWidth: string): string =>
  `calc(${controlWidth} + {spacing.${ROW_PADDING_X}} + ${CUE_WIDTH} + 2px)`;

/**
 * Which ends of the island's row hide content. Re-measured on scroll, whenever the scrollport or a control changes
 * size, and on any change inside the row: nested text can grow past a control that keeps its size (a counter gaining
 * a digit, a label overflowing its box). A measurement reads three layout values; subscribers hear only a change of
 * either edge.
 */
const createHiddenEdgesStore = () => {
  let scroller: HTMLElement | null = null;
  let hiddenEdges = NO_HIDDEN_EDGES;
  const listeners = new Set<() => void>();

  const measure = () => {
    if (!scroller) {
      return;
    }
    // Right-to-left rows scroll toward negative offsets; the distance from the start is the same either way.
    const offset = Math.abs(scroller.scrollLeft);
    const range = scroller.scrollWidth - scroller.clientWidth;
    const start = offset >= EDGE_TOLERANCE_PX;
    const end = range - offset >= EDGE_TOLERANCE_PX;

    if (start !== hiddenEdges.start || end !== hiddenEdges.end) {
      hiddenEdges = { end, start };
      listeners.forEach((listener) => listener());
    }
  };

  return {
    /**
     * Ref callback for the scroller. Controls that cannot shrink further overflow the scroller's box without resizing
     * it, so each control is observed too, including ones that mount later (a label whose chunk was still loading).
     * The cues take the pointer over the row's ends, so a wheel over them is forwarded to the row from the island.
     */
    attach: (element: HTMLElement | null) => {
      if (!element) {
        return;
      }
      scroller = element;
      const releaseWheel = wheelScrollsHorizontally(element);
      const island = element.parentElement;
      const forwardCueWheel = (event: WheelEvent) => {
        if (event.target instanceof Node && !element.contains(event.target)) {
          forwardWheelToStrip(element, event);
        }
      };
      island?.addEventListener('wheel', forwardCueWheel, { passive: false });
      const resizes = new ResizeObserver(measure);
      resizes.observe(element);
      for (const child of element.children) {
        resizes.observe(child);
      }
      const mutations = new MutationObserver((records) => {
        for (const record of records) {
          if (record.target === element) {
            record.removedNodes.forEach((node) => node instanceof Element && resizes.unobserve(node));
            record.addedNodes.forEach((node) => node instanceof Element && resizes.observe(node));
          }
        }
        measure();
      });
      mutations.observe(element, { characterData: true, childList: true, subtree: true });
      element.addEventListener('scroll', measure, { passive: true });

      return () => {
        if (typeof releaseWheel === 'function') {
          releaseWheel();
        }
        island?.removeEventListener('wheel', forwardCueWheel);
        resizes.disconnect();
        mutations.disconnect();
        element.removeEventListener('scroll', measure);
        scroller = null;
      };
    },
    getSnapshot: () => hiddenEdges,
    scrollTo: (edge: Edge) => {
      if (!scroller) {
        return;
      }
      const range = scroller.scrollWidth - scroller.clientWidth;
      const end = getComputedStyle(scroller).direction === 'rtl' ? -range : range;
      scroller.scrollTo({ behavior: prefersReducedMotion() ? 'instant' : 'smooth', left: edge === 'start' ? 0 : end });
    },
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
};

/**
 * A floating island of center chrome. Its row scrolls sideways, without a scrollbar, once the island reaches the width
 * its parent allows: by wheel, swipe, or focus, which brings a focused control clear of the cues. Each end hiding
 * content shows a fade into the island's background and a button that scrolls to that end.
 */
export const ChromeIsland = ({
  children,
  flexShrink = 1,
  maxW,
}: {
  children: ReactNode;
  flexShrink?: 0 | 1;
  maxW?: string;
}) => {
  const [{ attach, getSnapshot, scrollTo, subscribe }] = useState(createHiddenEdgesStore);
  const hiddenEdges = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);

  return (
    <Flex
      align="center"
      bg="bg.subtle"
      borderColor="border.subtle"
      borderWidth="1px"
      data-center-chrome-island=""
      flexShrink={flexShrink}
      h="8"
      maxW={maxW}
      minW="0"
      pointerEvents="auto"
      position="relative"
      rounded="md"
      shadow="sm"
    >
      {/* Labels truncate down to their own minimum widths first; past those the row overflows and scrolls. */}
      <HStack
        ref={attach}
        css={HIDDEN_SCROLLBAR_CSS}
        flex="1"
        gap="0.5"
        // Reaches 2px past the island's border above and below: a scroller clips, and a focused control needs room for
        // its ring.
        h="calc(100% + 4px)"
        // Contains the controls' own z-indices, so the cues after the row always paint and take the pointer above them.
        isolation="isolate"
        minW="0"
        overflowX="auto"
        overflowY="hidden"
        // A swipe on the row scrolls it and nothing behind it, even at either end.
        overscrollBehaviorX="contain"
        px={ROW_PADDING_X}
        // Focus scrolls a control clear of the cue covering that end, not merely onto the island's edge.
        scrollPaddingInline={CUE_WIDTH}
        touchAction="pan-x"
      >
        {children}
      </HStack>
      {hiddenEdges.start ? <OverflowCue edge="start" onScrollTo={scrollTo} /> : null}
      {hiddenEdges.end ? <OverflowCue edge="end" onScrollTo={scrollTo} /> : null}
    </Flex>
  );
};

/**
 * The fade and its button. The whole cue takes the pointer, so the half-hidden controls under the fade neither take a
 * click nor show hover; only the button acts.
 */
const OverflowCue = ({ edge, onScrollTo }: { edge: Edge; onScrollTo: (edge: Edge) => void }) => {
  const { t } = useTranslation();
  const label = edge === 'start' ? t('common.scrollToStart') : t('common.scrollToEnd');
  const scrollToEdge = useCallback(() => onScrollTo(edge), [edge, onScrollTo]);

  return (
    <Flex
      align="center"
      // Inside the border, so the fade rounds with the island's inner corners.
      borderEndRadius={edge === 'end' ? 'calc({radii.md} - 1px)' : undefined}
      borderStartRadius={edge === 'start' ? 'calc({radii.md} - 1px)' : undefined}
      css={CUE_CSS[edge]}
      data-overflow-cue={edge}
      insetInlineEnd={edge === 'end' ? '0' : undefined}
      insetInlineStart={edge === 'start' ? '0' : undefined}
      insetY="0"
      justify={edge === 'start' ? 'flex-start' : 'flex-end'}
      position="absolute"
      px="1"
      w={CUE_WIDTH}
      onMouseDown={keepFocus}
    >
      <Tooltip content={label}>
        <IconButton
          aria-label={label}
          color="fg.muted"
          size="sm"
          // Out of the tab order like other pointer shortcuts: focus already brings every control in the row into view.
          tabIndex={-1}
          variant="ghost"
          onClick={scrollToEdge}
        >
          <Icon as={edge === 'start' ? ChevronLeftIcon : ChevronRightIcon} boxSize="3.5" css={MIRRORED_IN_RTL} />
        </IconButton>
      </Tooltip>
    </Flex>
  );
};
