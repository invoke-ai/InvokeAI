import { Box, Code, HStack, Icon, Popover, Portal, Stack, Text } from '@chakra-ui/react';
import { describeDateRange, findInvalidDateToken, formatIsoDate, parseDateTokens } from '@platform/search/dateTokens';
import { CloseButton, IconButton } from '@platform/ui/Button';
import { CircleHelpIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { GallerySearchField } from './GallerySearchField';
import { useGalleryWidget } from './GalleryWidgetContext';

const SEARCH_DATE_HINT_ID = 'gallery-search-date-hint';
const HELP_POSITIONING = { placement: 'bottom-end' } as const;

/** The `key:value` forms `parseDateTokens` accepts, as shown in the help popover. */
const DATE_TOKEN_EXAMPLES = [
  { descriptionKey: 'widgets.gallery.searchHelpFrom', key: 'from:2026-07-14' },
  { descriptionKey: 'widgets.gallery.searchHelpTo', key: 'to:yesterday' },
  { descriptionKey: 'widgets.gallery.searchHelpDate', key: 'date:today' },
  { descriptionKey: 'widgets.gallery.searchHelpRelative', key: 'from:7d' },
] as const;

export const GalleryItemSearch = () => {
  const { i18n, t } = useTranslation();
  const { actions, gallery } = useGalleryWidget();

  const handleClearSearch = useCallback(() => actions.setSearchTerm(''), [actions]);

  // Valid tokens are legible as chips in the field itself, so only the failure
  // case still needs words — and it is positioned out of flow. As a sibling it
  // grew the field's row and knocked the wide header out of vertical centre.
  const invalidHint = useMemo(() => {
    const parse = parseDateTokens(gallery.searchTerm);
    const invalid = findInvalidDateToken(gallery.searchTerm, parse);

    return invalid ? t('widgets.gallery.dateFilterInvalid', { value: invalid.raw }) : null;
  }, [gallery.searchTerm, t]);

  // The chips show which text is a filter, not what it resolved to, so the
  // resolved range stays available to assistive tech.
  const appliedRange = useMemo(() => {
    const parse = parseDateTokens(gallery.searchTerm);
    const shape = parse.range ? describeDateRange(parse.range) : null;

    if (!shape) {
      return null;
    }

    const locale = i18n.language;

    switch (shape.kind) {
      case 'day':
        return t('widgets.gallery.dateFilterDay', { date: formatIsoDate(shape.date, locale) });
      case 'range':
        return t('widgets.gallery.dateFilterRange', {
          from: formatIsoDate(shape.from, locale),
          to: formatIsoDate(shape.to, locale),
        });
      case 'from':
        return t('widgets.gallery.dateFilterFrom', { date: formatIsoDate(shape.date, locale) });
      case 'through':
        return t('widgets.gallery.dateFilterThrough', { date: formatIsoDate(shape.date, locale) });
    }
  }, [gallery.searchTerm, i18n.language, t]);

  const endElement = useMemo(
    () => (
      <HStack flexShrink={0} gap="0">
        {gallery.searchTerm ? (
          <CloseButton aria-label={t('common.clearSearch')} size="2xs" onClick={handleClearSearch} />
        ) : null}
        <GallerySearchHelp />
      </HStack>
    ),
    [gallery.searchTerm, handleClearSearch, t]
  );

  return (
    <Box minW="0" position="relative" w="full">
      <GallerySearchField
        ariaLabel={t('widgets.gallery.searchImagesAriaLabel')}
        describedById={invalidHint ? SEARCH_DATE_HINT_ID : undefined}
        endElement={endElement}
        isInvalid={invalidHint !== null}
        placeholder={t('widgets.gallery.searchImagesPlaceholder')}
        value={gallery.searchTerm}
        onChange={actions.setSearchTerm}
      />
      {invalidHint ? (
        <Text
          color="fg.error"
          fontSize="2xs"
          id={SEARCH_DATE_HINT_ID}
          insetInlineStart="0"
          // Out of flow and inert: it must never shift the header row, nor
          // swallow clicks meant for the grid it now floats over.
          pointerEvents="none"
          position="absolute"
          role="status"
          top="100%"
        >
          {invalidHint}
        </Text>
      ) : null}
      {appliedRange ? (
        <Text role="status" srOnly>
          {appliedRange}
        </Text>
      ) : null}
    </Box>
  );
};

/** Documents the closed date-token grammar the search box accepts. */
const GallerySearchHelp = () => {
  const { t } = useTranslation();

  return (
    <Popover.Root positioning={HELP_POSITIONING}>
      <Popover.Trigger asChild>
        <IconButton aria-label={t('widgets.gallery.searchHelpTitle')} color="fg.subtle" size="2xs" variant="ghost">
          <Icon as={CircleHelpIcon} boxSize="3.5" />
        </IconButton>
      </Popover.Trigger>
      <Portal>
        <Popover.Positioner>
          <Popover.Content maxW="18rem" p="3">
            <Stack gap="2">
              <Text fontSize="xs" fontWeight="600">
                {t('widgets.gallery.searchHelpTitle')}
              </Text>
              <Text color="fg.muted" fontSize="2xs">
                {t('widgets.gallery.searchHelpIntro')}
              </Text>
              <Stack gap="1.5">
                {DATE_TOKEN_EXAMPLES.map(({ descriptionKey, key }) => (
                  <HStack key={key} align="start" gap="2">
                    <Code flexShrink={0} fontSize="2xs" px="1">
                      {key}
                    </Code>
                    <Text color="fg.muted" fontSize="2xs">
                      {t(descriptionKey)}
                    </Text>
                  </HStack>
                ))}
              </Stack>
            </Stack>
          </Popover.Content>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};
