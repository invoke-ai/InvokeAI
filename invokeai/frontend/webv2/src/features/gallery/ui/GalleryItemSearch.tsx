import type { GallerySemanticReference } from '@features/gallery/core/semanticImageQuery';

import { Box, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { semanticReferenceFromDataTransfer } from '@features/gallery/core/semanticImageQuery';
import { imageIndexAvailabilityOptions } from '@features/gallery/data/queries';
import { isImeComposing } from '@platform/browser/imeComposition';
import { useMountEffect } from '@platform/react/useMountEffect';
import { describeDateRange, findInvalidDateToken, formatIsoDate, parseDateTokens } from '@platform/search/dateTokens';
import { CloseButton, ToggleIconButton } from '@platform/ui/Button';
import { InputShell } from '@platform/ui/InputShell';
import { useQuery } from '@tanstack/react-query';
import { ImageIcon, LibraryBigIcon, MapIcon, SparklesIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';

import { GALLERY_SEMANTIC_SEARCH_DROP_ID, useGalleryImageDroppable } from './galleryDnd';
import { GallerySearchField } from './GallerySearchField';
import { GallerySearchHelp } from './GallerySearchHelp';
import { useGalleryUi } from './GalleryUiContext';
import { useGalleryWidget } from './GalleryWidgetContext';

const SEARCH_HINT_ID = 'gallery-search-hint';
/** Each semantic commit embeds and reranks on the server. */
export const SEMANTIC_SEARCH_COMMIT_DEBOUNCE_MS = 300;

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
  const { notifications } = useGalleryUi();
  const { data: indexAvailability } = useQuery(imageIndexAvailabilityOptions());
  const isSemanticMode = gallery.semanticSearchText !== null;
  const semanticText = gallery.semanticSearchText ?? '';
  // A persisted mode keeps the toggle reachable so users can exit.
  const isIndexConfigured = indexAvailability !== undefined && indexAvailability.state !== 'disabled';
  const showSemanticToggle = isIndexConfigured || isSemanticMode;

  // A timer that outlives the field is cleared; one that outlives its text is refused by the reducer.
  const inputRef = useRef<HTMLInputElement>(null);
  const commitTimerRef = useRef<number | null>(null);
  const cancelPendingCommit = useCallback(() => {
    if (commitTimerRef.current !== null) {
      window.clearTimeout(commitTimerRef.current);
      commitTimerRef.current = null;
    }
  }, []);

  useMountEffect(() => cancelPendingCommit);

  const handleChange = useCallback(
    (value: string) => {
      if (!isSemanticMode) {
        actions.setSearchTerm(value);
        return;
      }

      actions.setSemanticSearchText(value);
      cancelPendingCommit();
      // Commit even without the model: listing errors explain the unavailable search.
      commitTimerRef.current = window.setTimeout(() => {
        commitTimerRef.current = null;
        actions.commitSemanticSearch(value);
      }, SEMANTIC_SEARCH_COMMIT_DEBOUNCE_MS);
    },
    [actions, cancelPendingCommit, isSemanticMode]
  );

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (!isSemanticMode || event.key !== 'Enter' || isImeComposing(event.nativeEvent)) {
        return;
      }

      event.preventDefault();
      cancelPendingCommit();
      actions.commitSemanticSearch(event.currentTarget.value);
    },
    [actions, cancelPendingCommit, isSemanticMode]
  );

  const handleToggleSemanticMode = useCallback(
    (enabled: boolean) => {
      cancelPendingCommit();
      // Rendered before focus moves: a name change on a focused element is not read out again.
      flushSync(() => actions.setSemanticSearchMode(enabled));

      if (!enabled) {
        return;
      }

      inputRef.current?.focus();

      if (indexAvailability?.state === 'model_missing') {
        notifications.add({
          kind: 'info',
          message: t('widgets.gallery.semanticSearchInstallModel', { model: indexAvailability.modelName ?? '' }),
          title: t('widgets.gallery.semanticSearchModelMissingTitle'),
        });
      }
    },
    [actions, cancelPendingCommit, indexAvailability, notifications, t]
  );

  const handleClearSearch = useCallback(() => {
    cancelPendingCommit();
    actions.clearSearch();
  }, [actions, cancelPendingCommit]);
  const handleClearSemantic = useCallback(() => actions.setSemanticImageQuery(null), [actions]);
  const handleSearchScopeChange = useCallback(
    (searchAllBoards: boolean) => actions.setSearchScope(searchAllBoards ? 'all' : 'board'),
    [actions]
  );
  const isSearchingAllBoards = gallery.searchScope === 'all';
  // Inside the search row, so choosing a scope never moves the grid.
  const scopeToggle = useMemo(
    () =>
      gallery.isSearchActive ? (
        <ToggleIconButton
          checked={isSearchingAllBoards}
          color={isSearchingAllBoards ? 'accent.fg' : 'fg.muted'}
          data-gallery-search-scope=""
          icon={LibraryBigIcon}
          label={t('widgets.gallery.searchScopeAll')}
          tooltip={t('widgets.gallery.searchScopeCurrent', {
            scope: isSearchingAllBoards ? t('widgets.gallery.searchScopeAll') : t('widgets.gallery.searchScopeBoard'),
          })}
          variant="ghost"
          onCheckedChange={handleSearchScopeChange}
        />
      ) : null,
    [gallery.isSearchActive, handleSearchScopeChange, isSearchingAllBoards, t]
  );

  // In-app drags resolve through GalleryBoardDragMonitor; this only registers the target and reports hover.
  const { isOver: isItemDragOver, setNodeRef: setSemanticDropRef } = useGalleryImageDroppable({
    id: GALLERY_SEMANTIC_SEARCH_DROP_ID,
  });

  // Native drops handle OS files and external image URLs outside dnd-kit.
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

      // Plain text falls through to the browser's default input insertion.
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

  // Out of flow so it cannot enlarge or misalign the search row.
  const invalidHint = useMemo(() => {
    if (isSemanticMode) {
      return null;
    }

    const parse = parseDateTokens(gallery.searchTerm);
    const invalid = findInvalidDateToken(gallery.searchTerm, parse);

    return invalid ? t('widgets.gallery.dateFilterInvalid', { value: invalid.raw }) : null;
  }, [gallery.searchTerm, isSemanticMode, t]);

  // The index leaves archived boards out, so a ranked search there can only come back empty; clusters are exempt.
  const isRankedSearch =
    isSemanticMode || (gallery.semanticImageQuery !== null && gallery.semanticImageQuery.kind !== 'cluster');
  const isArchivedBoard =
    gallery.searchScope === 'board' &&
    gallery.boards.some((board) => board.id === gallery.selectedBoardId && board.archived);
  const semanticHint =
    isSemanticMode && indexAvailability?.state === 'model_missing'
      ? t('widgets.gallery.semanticSearchModelMissing', { model: indexAvailability.modelName ?? '' })
      : isRankedSearch && isArchivedBoard
        ? t('widgets.gallery.semanticSearchArchivedBoard')
        : null;

  const hint = invalidHint ?? semanticHint;

  // The chips show the filter text; assistive tech gets the resolved range.
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

  // One name in both states; the pressed state says which.
  const semanticToggle = useMemo(
    () =>
      showSemanticToggle ? (
        <ToggleIconButton
          checked={isSemanticMode}
          color={isSemanticMode ? 'fg.warning' : 'fg.muted'}
          icon={SparklesIcon}
          label={t('widgets.gallery.semanticSearchAriaLabel')}
          tooltip={isSemanticMode ? t('widgets.gallery.exitSemanticSearch') : t('widgets.gallery.searchSemantically')}
          variant="ghost"
          onCheckedChange={handleToggleSemanticMode}
        />
      ) : null,
    [handleToggleSemanticMode, isSemanticMode, showSemanticToggle, t]
  );

  const chipToggles = useMemo(
    () => (
      <>
        {scopeToggle}
        {semanticToggle}
      </>
    ),
    [scopeToggle, semanticToggle]
  );

  const endElement = useMemo(
    () => (
      <HStack flexShrink={0} gap="0">
        {scopeToggle}
        {semanticToggle}
        {isSemanticMode || gallery.searchTerm ? (
          <CloseButton
            aria-label={isSemanticMode ? t('widgets.gallery.clearSemanticSearch') : t('common.clearSearch')}
            size="sm"
            onClick={handleClearSearch}
          />
        ) : null}
        {/* The help documents metadata grammar. */}
        {isSemanticMode ? null : <GallerySearchHelp />}
      </HStack>
    ),
    [gallery.searchTerm, handleClearSearch, isSemanticMode, scopeToggle, semanticToggle, t]
  );

  return (
    <Box
      ref={setSemanticDropRef}
      minW="0"
      outline={isDropTargetActive ? '2px solid' : undefined}
      outlineColor={isDropTargetActive ? 'accent.solid' : undefined}
      position="relative"
      rounded="control"
      w="full"
      onDragLeave={handleNativeDragLeave}
      onDragOver={handleNativeDragOver}
      onDrop={handleNativeDrop}
    >
      <Stack gap="1" minW="0">
        {/* Image/file/cluster chips replace semantic text mode. */}
        {!isSemanticMode && gallery.semanticImageQuery ? (
          <GallerySemanticChip
            reference={gallery.semanticImageQuery}
            toggle={chipToggles}
            onClear={handleClearSemantic}
          />
        ) : (
          <GallerySearchField
            ref={inputRef}
            ariaLabel={
              isSemanticMode ? t('widgets.gallery.semanticSearchAriaLabel') : t('widgets.gallery.searchImagesAriaLabel')
            }
            describedById={hint ? SEARCH_HINT_ID : undefined}
            endElement={endElement}
            isInvalid={invalidHint !== null}
            mode={isSemanticMode ? 'semantic' : 'metadata'}
            placeholder={
              isSemanticMode
                ? t('widgets.gallery.semanticSearchPlaceholder')
                : t('widgets.gallery.searchImagesPlaceholder')
            }
            value={isSemanticMode ? semanticText : gallery.searchTerm}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
          />
        )}
      </Stack>
      {hint ? (
        <Text
          color={invalidHint ? 'fg.error' : 'fg.warning'}
          fontSize="xs"
          id={SEARCH_HINT_ID}
          insetInlineStart="0"
          // Inert: never shift the header row or swallow clicks meant for the grid.
          pointerEvents="none"
          position="absolute"
          role="status"
          top="100%"
        >
          {hint}
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

const GallerySemanticChip = ({
  onClear,
  reference,
  toggle,
}: {
  onClear: () => void;
  reference: GallerySemanticReference;
  toggle: ReactNode;
}) => {
  const { t } = useTranslation();
  const isText = reference.kind === 'text';
  const isCluster = reference.kind === 'cluster';
  const name = getSemanticReferenceName(reference) || t('widgets.gallery.semanticWebImage');
  const endElement = useMemo(
    () => (
      <HStack flexShrink={0} gap="0">
        {toggle}
        <CloseButton
          aria-label={
            isText
              ? t('widgets.gallery.clearSemanticSearch')
              : isCluster
                ? t('widgets.gallery.clearClusterSearch')
                : t('widgets.gallery.clearImageSearch')
          }
          size="sm"
          onClick={onClear}
        />
      </HStack>
    ),
    [isCluster, isText, onClear, t, toggle]
  );
  const kindIcon = useMemo(
    () => (
      <Icon
        as={isText ? SparklesIcon : isCluster ? MapIcon : ImageIcon}
        boxSize="3.5"
        color="fg.subtle"
        flexShrink={0}
      />
    ),
    [isCluster, isText]
  );

  return (
    <InputShell endElement={endElement} startElement={kindIcon} title={getSemanticReferenceTitle(reference)}>
      <Text color="fg.muted" flex="1" fontSize="md" minW="0" truncate>
        {isText
          ? t('widgets.gallery.semanticTextSearch', { name })
          : isCluster
            ? t('widgets.gallery.semanticCluster', { name })
            : t('widgets.gallery.semanticSimilarTo', { name })}
      </Text>
    </InputShell>
  );
};
