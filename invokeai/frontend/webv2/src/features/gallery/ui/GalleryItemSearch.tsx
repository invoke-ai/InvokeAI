import type { GallerySemanticReference } from '@features/gallery/core/semanticImageQuery';

import { Box, Code, HStack, Icon, Popover, Portal, Stack, Text } from '@chakra-ui/react';
import { semanticReferenceFromDataTransfer } from '@features/gallery/core/semanticImageQuery';
import { describeDateRange, findInvalidDateToken, formatIsoDate, parseDateTokens } from '@platform/search/dateTokens';
import { CloseButton, IconButton } from '@platform/ui/Button';
import { CircleHelpIcon, ImageIcon, MapIcon, SparklesIcon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { GALLERY_SEMANTIC_SEARCH_DROP_ID, useGalleryImageDroppable } from './galleryDnd';
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

/** The full identity of the reference, surfaced as the chip's hover title. */
const getSemanticReferenceTitle = (reference: GallerySemanticReference): string => {
  switch (reference.kind) {
    case 'text':
      return reference.query;
    case 'image':
      return reference.imageName;
    case 'url':
      return reference.url;
    case 'file':
      return reference.label;
    case 'cluster':
      return reference.label;
  }
};

/** A short human-readable name for the chip; empty when nothing legible exists. */
const getSemanticReferenceName = (reference: GallerySemanticReference): string => {
  switch (reference.kind) {
    case 'text':
      return reference.query;
    case 'image':
      return reference.imageName;
    case 'file':
      return reference.label;
    case 'cluster':
      return reference.label;
    case 'url': {
      try {
        const parsed = new URL(reference.url);

        return decodeURIComponent(parsed.pathname.split('/').pop() || parsed.hostname);
      } catch {
        return '';
      }
    }
  }
};

export const GalleryItemSearch = () => {
  const { i18n, t } = useTranslation();
  const { actions, gallery } = useGalleryWidget();

  const handleClearSearch = useCallback(() => actions.setSearchTerm(''), [actions]);
  const handleClearSemantic = useCallback(() => actions.setSemanticImageQuery(null), [actions]);

  // Date tokens are metadata-search grammar; a semantic query is free text,
  // so the whole field content is taken verbatim.
  const semanticQueryText = gallery.searchTerm.trim();
  const handleSemanticSearch = useCallback(() => {
    const query = semanticQueryText;

    if (query) {
      actions.setSemanticImageQuery({ kind: 'text', query });
    }
  }, [actions, semanticQueryText]);

  // In-app drags (dnd-kit) resolve through GalleryBoardDragMonitor; this hook
  // only registers the drop target and reports hover for the highlight.
  const { isOver: isItemDragOver, setNodeRef: setSemanticDropRef } = useGalleryImageDroppable({
    id: GALLERY_SEMANTIC_SEARCH_DROP_ID,
  });

  // Native HTML5 drops reach here for anything dnd-kit does not manage: files
  // from the OS and images dragged in from other pages (URL drags).
  const [isNativeDropOver, setIsNativeDropOver] = useState(false);

  const handleNativeDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    const types = event.dataTransfer.types;

    if (types.includes('Files') || types.includes('text/uri-list')) {
      event.preventDefault();
      setIsNativeDropOver(true);
    }
  }, []);

  const handleNativeDragLeave = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
      setIsNativeDropOver(false);
    }
  }, []);

  const handleNativeDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      const types = event.dataTransfer.types;

      // Only claim drops this feature understands; a plain-text drop must
      // fall through to the browser's default insertion into the input.
      if (!types.includes('Files') && !types.includes('text/uri-list')) {
        setIsNativeDropOver(false);
        return;
      }

      event.preventDefault();
      setIsNativeDropOver(false);

      const reference = semanticReferenceFromDataTransfer(event.dataTransfer);

      if (reference) {
        actions.setSemanticImageQuery(reference);
      }
    },
    [actions]
  );

  const isDropTargetActive = isItemDragOver || isNativeDropOver;

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
        {semanticQueryText ? (
          <IconButton
            aria-label={t('widgets.gallery.searchSemantically')}
            color="fg.muted"
            size="2xs"
            title={t('widgets.gallery.searchSemantically')}
            variant="ghost"
            onClick={handleSemanticSearch}
          >
            <Icon as={SparklesIcon} boxSize="3.5" />
          </IconButton>
        ) : null}
        {gallery.searchTerm ? (
          <CloseButton aria-label={t('common.clearSearch')} size="2xs" onClick={handleClearSearch} />
        ) : null}
        <GallerySearchHelp />
      </HStack>
    ),
    [gallery.searchTerm, handleClearSearch, handleSemanticSearch, semanticQueryText, t]
  );

  return (
    <Box
      ref={setSemanticDropRef}
      minW="0"
      outline={isDropTargetActive ? '2px solid' : undefined}
      outlineColor={isDropTargetActive ? 'accent.solid' : undefined}
      position="relative"
      rounded="md"
      w="full"
      onDragLeave={handleNativeDragLeave}
      onDragOver={handleNativeDragOver}
      onDrop={handleNativeDrop}
    >
      {gallery.semanticImageQuery ? (
        <GallerySemanticChip reference={gallery.semanticImageQuery} onClear={handleClearSemantic} />
      ) : (
        <GallerySearchField
          ariaLabel={t('widgets.gallery.searchImagesAriaLabel')}
          describedById={invalidHint ? SEARCH_DATE_HINT_ID : undefined}
          endElement={endElement}
          isInvalid={invalidHint !== null}
          placeholder={t('widgets.gallery.searchImagesPlaceholder')}
          value={gallery.searchTerm}
          onChange={actions.setSearchTerm}
        />
      )}
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

/**
 * The active semantic query — a text prompt or an image-similarity
 * reference — rendered in place of the search input. Clearing it restores
 * metadata search.
 */
const GallerySemanticChip = ({ onClear, reference }: { onClear: () => void; reference: GallerySemanticReference }) => {
  const { t } = useTranslation();
  const isText = reference.kind === 'text';
  const isCluster = reference.kind === 'cluster';
  const name = getSemanticReferenceName(reference) || t('widgets.gallery.semanticWebImage');

  return (
    <HStack
      bg="bg"
      borderColor="border.emphasized"
      borderWidth="1px"
      gap="1.5"
      h="8"
      minW="0"
      px="2"
      rounded="md"
      title={getSemanticReferenceTitle(reference)}
      w="full"
    >
      <Icon
        as={isText ? SparklesIcon : isCluster ? MapIcon : ImageIcon}
        boxSize="3.5"
        color="fg.subtle"
        flexShrink={0}
      />
      <Text color="fg.muted" flex="1" fontSize="xs" minW="0" truncate>
        {isText
          ? t('widgets.gallery.semanticTextSearch', { name })
          : isCluster
            ? t('widgets.gallery.semanticCluster', { name })
            : t('widgets.gallery.semanticSimilarTo', { name })}
      </Text>
      <CloseButton
        aria-label={
          isText
            ? t('widgets.gallery.clearSemanticSearch')
            : isCluster
              ? t('widgets.gallery.clearClusterSearch')
              : t('widgets.gallery.clearImageSearch')
        }
        size="2xs"
        onClick={onClear}
      />
    </HStack>
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
