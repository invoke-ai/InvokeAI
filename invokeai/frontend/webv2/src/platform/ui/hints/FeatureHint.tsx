import type { ReactElement } from 'react';

import { Heading, HoverCard, HStack, Icon, Link, Portal, Separator, Spacer, Stack, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { ExternalLinkIcon } from 'lucide-react';
import { useMemo } from 'react';
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
              <Button color="fg.subtle" fontSize="2xs" h="auto" px="0" size="2xs" variant="plain" onClick={onDisable}>
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

export interface FeatureHintProps {
  hint: FeatureHintId;
  /** Becomes the hover target verbatim — it gains data attributes, not a wrapper. */
  children: ReactElement;
}

/**
 * Attaches an informational hint card to a control, opened by hovering the
 * child. Copy lives in the `hints.*` i18n namespace; {@link getFeatureHint}
 * supplies the optional support link.
 *
 * The child is the trigger via `asChild`, and zag builds trigger props with
 * `normalize.element` rather than `normalize.button` — so wrapping a field
 * label adds pointer/focus handlers and `data-*` attributes but no `tabindex`
 * and no role. Labels stay out of the tab order and keep click-to-focus.
 *
 * When hints are turned off the child renders untouched, with no popper mounted.
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
      <HoverCard.Trigger asChild>{children}</HoverCard.Trigger>
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
