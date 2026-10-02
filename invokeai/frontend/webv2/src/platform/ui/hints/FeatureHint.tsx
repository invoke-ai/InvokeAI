import type { HTMLAttributes, ReactElement, ReactNode } from 'react';

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
import { Button } from '@platform/ui/Button';
import { ExternalLinkIcon } from 'lucide-react';
import { cloneElement, createContext, useContext, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { DEFAULT_HINT_PLACEMENT, getFeatureHint, type FeatureHintId } from './hintRegistry';
import { useFeatureHints } from './hintsContext';

/** Long enough that hints stay out of the way while a user works a panel. */
const OPEN_DELAY = 600;
/** Long enough to let the pointer travel from the label into the card's links. */
const CLOSE_DELAY = 300;

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
      <Heading fontSize="xs" fontWeight="600">
        {heading}
      </Heading>
      <Separator />
      {paragraphs.map((paragraph) => (
        <Text key={paragraph} color="fg.muted" fontSize="xs" lineHeight="1.45">
          {paragraph}
        </Text>
      ))}
      {(onDisable || href) && (
        <>
          <Separator />
          <HStack gap="2" minH="4">
            {onDisable && (
              <Button color="fg.subtle" fontSize="2xs" h="auto" px="1" size="2xs" variant="plain" onClick={onDisable}>
                {t('common.dontShowMeThese')}
              </Button>
            )}
            <Spacer />
            {href && (
              <Link fontSize="2xs" gap="1" href={href} rel="noreferrer" target="_blank">
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
  const positioning = useMemo(
    () => ({ gutter: 8, placement: getFeatureHint(hint).placement ?? DEFAULT_HINT_PLACEMENT }),
    [hint]
  );

  if (!enabled) {
    return children;
  }

  return (
    <HoverCard.Root closeDelay={CLOSE_DELAY} lazyMount openDelay={OPEN_DELAY} positioning={positioning} unmountOnExit>
      <ExclusionProvider>
        <HoverCard.Trigger asChild>{children}</HoverCard.Trigger>
      </ExclusionProvider>
      <Portal>
        <HoverCard.Positioner>
          <HoverCard.Content>
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
