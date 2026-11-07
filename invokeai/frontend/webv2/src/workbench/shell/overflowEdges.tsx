import type { MouseEvent } from 'react';

import { Flex, Icon, Text } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { forwardWheelToStrip, wheelScrollsHorizontally } from '@platform/ui/wheelScrollsHorizontally';
import { ChevronLeftIcon, ChevronRightIcon } from 'lucide-react';
import { useCallback, useState, useSyncExternalStore } from 'react';

/**
 * A row of top or center chrome that scrolls sideways without a scrollbar, and the cues at the ends that hide content:
 * a fade into the chrome's background with a button that scrolls to that end. Shared by the center's chrome islands
 * and the layout preset strip.
 */

export type OverflowEdge = 'end' | 'start';
type Edge = OverflowEdge;

export interface HiddenEdges {
  end: boolean;
  /** Items (matching the store's `itemSelector`) lying wholly or partly past the end edge; 0 without a selector. */
  endCount: number;
  start: boolean;
}

const NO_HIDDEN_EDGES: HiddenEdges = { end: false, endCount: 0, start: false };

// A fractional zoom can leave a sub-pixel remainder at either end of a fully scrolled row; that hides nothing.
const EDGE_TOLERANCE_PX = 1;

/** Fade plus the cue's button and the island's inline padding beside it. */
export const CUE_WIDTH = '2.75rem';
/** The row's inline padding; the cue covers the end padding, the start padding stays in front of the first control. */
export const ROW_PADDING_X = '1';
const CUE_FADE = '1rem';

const fadeToward = (side: 'left' | 'right') =>
  `linear-gradient(to ${side}, transparent, {colors.bg.subtle} ${CUE_FADE})`;

// The fade runs from the row into the island's own background at the edge hiding content; right-to-left mirrors it.
const CUE_CSS: Record<Edge, object> = {
  end: { bgImage: fadeToward('right'), _rtl: { bgImage: fadeToward('left') } },
  start: { bgImage: fadeToward('left'), _rtl: { bgImage: fadeToward('right') } },
};

const MIRRORED_IN_RTL = { _rtl: { transform: 'scaleX(-1)' } } as const;

export const HIDDEN_SCROLLBAR_CSS = { '&::-webkit-scrollbar': { display: 'none' }, scrollbarWidth: 'none' } as const;

const prefersReducedMotion = (): boolean =>
  document.documentElement.dataset.reduceMotion === 'true' ||
  globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;

/**
 * Keep focus on the control the user was on, whether the press lands on the button or the fade: the cue is a pointer
 * shortcut, and focus already scrolls the row.
 */
const keepFocus = (event: MouseEvent) => event.preventDefault();

/**
 * Which ends of the island's row hide content. Re-measured on scroll, whenever the scrollport or a control changes
 * size, and on any change inside the row: nested text can grow past a control that keeps its size (a counter gaining
 * a digit, a label overflowing its box). A measurement reads three layout values; subscribers hear only a change of
 * either edge.
 */
const createHiddenEdgesStore = ({ itemSelector, keepInViewSelector }: OverflowEdgesOptions) => {
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
    const bounds = end && itemSelector ? scroller.getBoundingClientRect() : null;
    const isRtl = bounds !== null && getComputedStyle(scroller).direction === 'rtl';
    const endCount = bounds
      ? [...scroller.querySelectorAll(itemSelector!)].filter((item) => {
          const rect = item.getBoundingClientRect();

          return isRtl ? rect.left < bounds.left - EDGE_TOLERANCE_PX : rect.right > bounds.right + EDGE_TOLERANCE_PX;
        }).length
      : 0;

    if (start !== hiddenEdges.start || end !== hiddenEdges.end || endCount !== hiddenEdges.endCount) {
      hiddenEdges = { end, endCount, start };
      listeners.forEach((listener) => listener());
    }
  };

  /**
   * Scrolls the kept item back into view when the row clips it. Runs when the row resizes or the item changes, never on
   * scroll, so a row the user scrolled away stays where they put it.
   */
  const reveal = () => {
    const item = keepInViewSelector ? scroller?.querySelector<HTMLElement>(keepInViewSelector) : null;

    // A tab being dragged is not the user choosing a preset; the row stays where the drag holds it.
    if (!scroller || !item || scroller.querySelector('[aria-pressed="true"]')) {
      return;
    }

    const bounds = scroller.getBoundingClientRect();
    const rect = item.getBoundingClientRect();

    if (rect.left < bounds.left - EDGE_TOLERANCE_PX || rect.right > bounds.right + EDGE_TOLERANCE_PX) {
      item.scrollIntoView({ behavior: 'instant', block: 'nearest', inline: 'nearest' });
    }
  };

  return {
    /**
     * Ref callback for the scroller. Controls that cannot shrink further overflow the scroller's box without resizing
     * it, so each control is observed too, including ones that mount later (a label whose chunk was still loading).
     * The cues take the pointer over the row's ends, so a wheel over them is forwarded to the row from its parent, which
     * holds the cues.
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
      // Only the row itself narrowing or widening re-reveals the kept item; its controls resize for their own reasons
      // (a drag's measuring, a label loading), and revealing then would fight the user's own scrolling.
      let rowWidth = element.clientWidth;
      const resizes = new ResizeObserver(() => {
        if (element.clientWidth !== rowWidth) {
          rowWidth = element.clientWidth;
          reveal();
        }
        measure();
      });
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
        // A newly selected item is revealed; an unrelated change inside the row only re-measures.
        if (records.some((record) => record.type === 'attributes')) {
          reveal();
        }
        measure();
      });
      mutations.observe(element, {
        characterData: true,
        childList: true,
        subtree: true,
        ...(keepInViewSelector ? { attributeFilter: ['aria-selected'], attributes: true } : {}),
      });
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

export interface OverflowEdgesOptions {
  /** Counts the items past the end, for a cue that says how many more there are. */
  itemSelector?: string;
  /** The item (such as the selected tab) brought back into view when the row resizes or the item changes. */
  keepInViewSelector?: string;
}

/** The scroller's ref callback, which ends hide content, and a scroll to either end. */
export const useOverflowEdges = (options: OverflowEdgesOptions = {}) => {
  const [{ attach, getSnapshot, scrollTo, subscribe }] = useState(() => createHiddenEdgesStore(options));
  const hiddenEdges = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);

  return { attach, hiddenEdges, scrollTo };
};

/**
 * The fade and its button. The whole cue takes the pointer, so the half-hidden controls under the fade neither take a
 * click nor show hover; only the button acts.
 */
export const OverflowCue = ({
  ariaLabel,
  count,
  edge,
  label,
  onScrollTo,
}: {
  /** The button's name, when it says more than the tooltip (how many items it reveals). */
  ariaLabel?: string;
  /** Items the button reveals, shown as "+N" beside the chevron. */
  count?: number;
  edge: Edge;
  label: string;
  onScrollTo: (edge: Edge) => void;
}) => {
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
      // A count shares the cue's width with the chevron, so a narrow row keeps as much of itself in view.
      w={CUE_WIDTH}
      onMouseDown={keepFocus}
    >
      <Tooltip content={label}>
        {count ? (
          <Button
            aria-label={ariaLabel ?? label}
            color="fg.muted"
            gap="0"
            minW="0"
            px="0.5"
            size="sm"
            tabIndex={-1}
            variant="ghost"
            onClick={scrollToEdge}
          >
            <Text aria-hidden as="span" fontSize="xs" fontVariantNumeric="tabular-nums">
              +{count}
            </Text>
            <Icon as={edge === 'start' ? ChevronLeftIcon : ChevronRightIcon} boxSize="3.5" css={MIRRORED_IN_RTL} />
          </Button>
        ) : (
          <IconButton
            aria-label={ariaLabel ?? label}
            color="fg.muted"
            size="sm"
            // Out of the tab order like other pointer shortcuts: focus already brings every control in the row into view.
            tabIndex={-1}
            variant="ghost"
            onClick={scrollToEdge}
          >
            <Icon as={edge === 'start' ? ChevronLeftIcon : ChevronRightIcon} boxSize="3.5" css={MIRRORED_IN_RTL} />
          </IconButton>
        )}
      </Tooltip>
    </Flex>
  );
};
