import type { HTMLAttributes, PointerEvent, ReactElement, ReactNode } from 'react';

import {
  Heading,
  HoverCard,
  HStack,
  Icon,
  Link,
  mergeProps,
  Portal,
  Separator,
  Spacer,
  Stack,
  Text,
  useHoverCardContext,
} from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button } from '@platform/ui/Button';
import { ExternalLinkIcon } from 'lucide-react';
import { cloneElement, createContext, useCallback, useContext, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import {
  createGraceCorridor,
  findScrollingPanel,
  getHintAnchorRect,
  HINT_GUTTER_PX,
  HINT_VIEWPORT_PADDING_PX,
} from './hintGeometry';
import { DEFAULT_HINT_PLACEMENT, getFeatureHint, type FeatureHintId } from './hintRegistry';
import { useFeatureHints } from './hintsContext';

type HintPositioning = NonNullable<HoverCard.RootProps['positioning']>;

/** Long enough that hints stay out of the way while a user works a panel. */
const OPEN_DELAY = 600;
/** Forgives a brief slip off the label or card; the grace corridor covers the trip from label to card. */
const CLOSE_DELAY = 300;
/** How long the corridor waits for progress toward the card; resting on the way, or on the field's control, ends it. */
const CORRIDOR_GRACE_MS = 1000;
const HINT_TRIGGER_SELECTOR = '[data-scope="hover-card"][data-part="trigger"]';

const readParagraphs = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === 'string') : [];

const HintCard = ({ hint, onDisable }: { hint: FeatureHintId; onDisable: (() => void) | null }) => {
  const { t } = useTranslation();
  const { href } = getFeatureHint(hint);
  const heading = t(`hints.${hint}.heading`);
  const paragraphs = readParagraphs(t(`hints.${hint}.paragraphs`, { returnObjects: true }));

  // Padding belongs to the `hoverCard` content recipe, not here.
  return (
    <Stack gap="1.5">
      <Heading fontSize="md" fontWeight="600">
        {heading}
      </Heading>
      <Separator />
      {paragraphs.map((paragraph) => (
        <Text key={paragraph} color="fg.muted" fontSize="md" lineHeight="1.45">
          {paragraph}
        </Text>
      ))}
      {(onDisable || href) && (
        <>
          <Separator />
          <HStack gap="2" minH="4">
            {onDisable && (
              <Button color="fg.subtle" fontSize="xs" h="auto" px="1" size="sm" variant="plain" onClick={onDisable}>
                {t('common.dontShowMeThese')}
              </Button>
            )}
            <Spacer />
            {href && (
              <Link fontSize="xs" gap="1" href={href} rel="noreferrer" target="_blank">
                {t('common.learnMore')}
                <Icon as={ExternalLinkIcon} boxSize="3" />
              </Link>
            )}
          </HStack>
        </>
      )}
    </Stack>
  );
};

type ExclusionHandlers = Pick<HTMLAttributes<HTMLElement>, 'onPointerEnter' | 'onPointerLeave'>;

const ExclusionContext = createContext<ExclusionHandlers | null>(null);

/** Hands the trigger's own pointer handlers, swapped, to any exclusion inside the trigger. */
const ExclusionProvider = ({ children }: { children: ReactNode }) => {
  const { onPointerEnter, onPointerLeave } = useHoverCardContext().getTriggerProps();
  const handlers = useMemo<ExclusionHandlers | null>(
    () =>
      onPointerEnter && onPointerLeave ? { onPointerEnter: onPointerLeave, onPointerLeave: onPointerEnter } : null,
    [onPointerEnter, onPointerLeave]
  );
  return <ExclusionContext.Provider value={handlers}>{children}</ExclusionContext.Provider>;
};

/**
 * Suppress pointer hints over nested tooltip controls by treating entry/exit as trigger leave/enter; keyboard
 * focus is unaffected.
 */
export const FeatureHintExclusion = ({ children }: { children: ReactElement<HTMLAttributes<HTMLElement>> }) => {
  const handlers = useContext(ExclusionContext);
  return handlers ? cloneElement(children, mergeProps(handlers, children.props)) : children;
};

type Point = { x: number; y: number };

/**
 * What one hint's hover-card callbacks share outside render: the card element, the panel found once per opening,
 * and the grace corridor in progress.
 */
const createHintSession = () => {
  let card: HTMLElement | null = null;
  let panel: { panel: HTMLElement | null; trigger: Element } | null = null;
  let corridor: AbortController | null = null;

  const cancelCorridor = () => corridor?.abort();

  return {
    cancelCorridor,
    getAnchorRect: (anchor: unknown) => {
      if (!(anchor instanceof HTMLElement)) {
        return null;
      }

      if (panel?.trigger !== anchor) {
        panel = { panel: findScrollingPanel(anchor), trigger: anchor };
      }

      return getHintAnchorRect(anchor, panel.panel, card);
    },
    resetPanel: () => {
      panel = null;
    },
    setCard: (element: HTMLElement | null) => {
      card = element;
    },
    /**
     * Leaving the label toward an open card keeps it open while the pointer travels the corridor to it, as long as it
     * gets closer within each `CORRIDOR_GRACE_MS`. Leaving the corridor, another hint's label, a scroll, or the pointer leaving the
     * page calls `close`, as leaving the label would have. Returns false when no corridor applies.
     */
    startCorridor: (exit: Point, trigger: HTMLElement, close: () => void): boolean => {
      cancelCorridor();

      const cardRect = card?.getBoundingClientRect();
      const isInCorridor = cardRect ? createGraceCorridor(exit, trigger.getBoundingClientRect(), cardRect) : null;

      if (!card || !cardRect || !isInCorridor) {
        return false;
      }

      const openCard = card;
      const session = new AbortController();
      const listen = { capture: true, signal: session.signal };
      // Anything but steady travel to the card gives the card back to the ordinary close delay.
      const giveUp = () => {
        session.abort();
        close();
      };
      // The arrow lies outside the content rect, so the event target counts as well as the point.
      const isOverCard = (x: number, y: number, target: EventTarget | null) =>
        (x >= cardRect.left && x <= cardRect.right && y >= cardRect.top && y <= cardRect.bottom) ||
        (target instanceof Node && openCard.contains(target));
      const distanceToCard = (x: number) => Math.min(Math.abs(x - cardRect.left), Math.abs(x - cardRect.right));
      let closest = distanceToCard(exit.x);
      let graceTimer = globalThis.setTimeout(giveUp, CORRIDOR_GRACE_MS);

      corridor = session;
      session.signal.addEventListener('abort', () => globalThis.clearTimeout(graceTimer));
      document.addEventListener(
        'pointermove',
        ({ clientX: x, clientY: y, target }) => {
          // Over the card, or closed some other way: the hover card owns what happens next.
          if (!openCard.isConnected || isOverCard(x, y, target)) {
            session.abort();
          } else if (!isInCorridor({ x, y })) {
            giveUp();
          } else if (distanceToCard(x) < closest) {
            // Progress toward the card renews the grace window; resting or wandering lets it run out.
            closest = distanceToCard(x);
            globalThis.clearTimeout(graceTimer);
            graceTimer = globalThis.setTimeout(giveUp, CORRIDOR_GRACE_MS);
          }
        },
        listen
      );
      // Another hint's label takes over, so only one card is ever open.
      document.addEventListener(
        'pointerover',
        ({ target }) => {
          const otherTrigger = target instanceof Element ? target.closest(HINT_TRIGGER_SELECTOR) : null;

          if (otherTrigger && otherTrigger !== trigger) {
            giveUp();
          }
        },
        listen
      );
      // The card's rect is fixed for the trip; a scroll moves it.
      window.addEventListener('scroll', giveUp, listen);
      window.addEventListener('blur', giveUp, listen);
      // Not in capture: there, every descendant's pointerleave would pass through.
      document.documentElement.addEventListener('pointerleave', giveUp, { signal: session.signal });

      return true;
    },
  };
};

type HintCorridor = Pick<ReturnType<typeof createHintSession>, 'cancelCorridor' | 'startCorridor'>;

/**
 * The label as the card's trigger. The card opens beside the panel, out of reach of the close delay, so the trigger
 * hands an open card's pointer leave to the session's grace corridor. Escape and blur still close it through the
 * hover card.
 */
const HintTrigger = ({ cancelCorridor, children, startCorridor }: HintCorridor & { children: ReactElement }) => {
  const hoverCard = useHoverCardContext();
  const triggerProps = hoverCard.getTriggerProps();
  const { onPointerEnter, onPointerLeave } = triggerProps;

  const handlePointerEnter = (event: PointerEvent<HTMLElement>) => {
    cancelCorridor();
    onPointerEnter?.(event);
  };
  const handlePointerLeave = (event: PointerEvent<HTMLElement>) => {
    const close = () => onPointerLeave?.(event);
    // A held button is a drag, such as a scrubber taking pointer capture, not a trip to the card.
    const isHeld =
      hoverCard.open &&
      event.pointerType !== 'touch' &&
      event.buttons === 0 &&
      startCorridor({ x: event.clientX, y: event.clientY }, event.currentTarget, close);

    if (!isHeld) {
      close();
    }
  };

  return cloneElement(
    children,
    mergeProps(
      { ...triggerProps, onPointerEnter: handlePointerEnter, onPointerLeave: handlePointerLeave },
      children.props as Record<string, unknown>
    )
  );
};

export interface FeatureHintProps {
  hint: FeatureHintId;
  /** Becomes the hover target verbatim — it gains data attributes, not a wrapper. */
  children: ReactElement;
}

/**
 * Attach to the child without button roles/tabindex so labels retain semantics; disabled hints return the child
 * untouched.
 */
export const FeatureHint = ({ children, hint }: FeatureHintProps) => {
  const { enabled, onDisable } = useFeatureHints();
  const [{ cancelCorridor, getAnchorRect, resetPanel, setCard, startCorridor }] = useState(createHintSession);
  const positioning = useMemo<HintPositioning>(
    () => ({
      getAnchorRect,
      gutter: HINT_GUTTER_PX,
      overflowPadding: HINT_VIEWPORT_PADDING_PX,
      placement: getFeatureHint(hint).placement ?? DEFAULT_HINT_PLACEMENT,
    }),
    [getAnchorRect, hint]
  );
  // The panel around a label can change between openings, never during one. A press outside the card closes it
  // through the hover card's dismissal, which ends any corridor with it.
  const handleOpenChange = useCallback(
    ({ open }: { open: boolean }) => {
      if (open) {
        resetPanel();
      } else {
        cancelCorridor();
      }
    },
    [cancelCorridor, resetPanel]
  );

  useMountEffect(() => cancelCorridor);

  if (!enabled) {
    return children;
  }

  return (
    <HoverCard.Root
      closeDelay={CLOSE_DELAY}
      lazyMount
      openDelay={OPEN_DELAY}
      positioning={positioning}
      unmountOnExit
      onOpenChange={handleOpenChange}
    >
      <ExclusionProvider>
        <HintTrigger cancelCorridor={cancelCorridor} startCorridor={startCorridor}>
          {children}
        </HintTrigger>
      </ExclusionProvider>
      <Portal>
        <HoverCard.Positioner>
          <HoverCard.Content ref={setCard}>
            <HoverCard.Arrow>
              <HoverCard.ArrowTip />
            </HoverCard.Arrow>
            <HintCard hint={hint} onDisable={onDisable} />
          </HoverCard.Content>
        </HoverCard.Positioner>
      </Portal>
    </HoverCard.Root>
  );
};
