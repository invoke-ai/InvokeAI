import type { GalleryView } from '@features/gallery/core/types';

import {
  Box,
  HStack,
  Icon,
  Input,
  InputGroup,
  Menu,
  Portal,
  Slider,
  Spacer,
  Stack,
  Switch,
  Text,
} from '@chakra-ui/react';
import { isDateBoardId } from '@features/gallery/data/backend';
import {
  formatIsoDate,
  isPossibleDatePrefix,
  matchTrailingDateToken,
  parseDateTokens,
} from '@platform/search/dateTokens';
import { Button, CloseButton, IconButton, Tabs } from '@platform/ui';
import { SearchIcon, SettingsIcon, UploadIcon } from 'lucide-react';
import { useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { GalleryBoardSelect } from './GalleryBoardSelect';
import { useGalleryWidget } from './GalleryWidgetContext';

const ACCEPTED_UPLOAD_EXTENSIONS = 'image/png,image/jpeg,image/webp';
const galleryViewTabs = [
  { labelKey: 'common.images', value: 'images' },
  { labelKey: 'common.assets', value: 'assets' },
] satisfies { labelKey: string; value: GalleryView }[];

const UPLOAD_INPUT_STYLE = { display: 'none' } as const;
const GALLERY_SETTINGS_POSITIONING = { placement: 'bottom-end' } as const;
const SEARCH_START_ELEMENT = <Icon as={SearchIcon} size="sm" />;
const SEARCH_DATE_HINT_ID = 'gallery-search-date-hint';

export const GalleryToolbar = ({ layout }: { layout: 'stacked' | 'wide' }) => {
  const { i18n, t } = useTranslation();
  const { actions, gallery } = useGalleryWidget();
  const isWide = layout === 'wide';

  const handleViewChange = useCallback(
    (event: { value: string }) => actions.setView(event.value as GalleryView),
    [actions]
  );

  const handleClearSearch = useCallback(() => actions.setSearchTerm(''), [actions]);

  const handleSearchChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => actions.setSearchTerm(event.currentTarget.value),
    [actions]
  );

  const searchClearButton = useMemo(
    () =>
      gallery.searchTerm ? (
        <CloseButton size="2xs" aria-label={t('common.clearSearch')} onClick={handleClearSearch} me="-2" />
      ) : null,
    [gallery.searchTerm, handleClearSearch, t]
  );

  const viewTabs = useMemo(
    () => (
      <Tabs.Root size="sm" variant="subtle" value={gallery.galleryView} onValueChange={handleViewChange}>
        <Tabs.List>
          {galleryViewTabs.map((item) => (
            <Tabs.Trigger key={item.value} value={item.value} fontSize="xs">
              {t(item.labelKey)}
            </Tabs.Trigger>
          ))}
        </Tabs.List>
      </Tabs.Root>
    ),
    [gallery.galleryView, handleViewChange, t]
  );

  const toolbarActions = useMemo(
    () => (
      <HStack gap="2">
        <GalleryUploadButton />
        <GallerySettingsMenu />
      </HStack>
    ),
    []
  );

  // Derived from the raw search term: an applied-range summary, or invalid-
  // token feedback. A trailing token that could still become valid (`from:`,
  // `from:2026-`) is normal typing, not an error.
  const dateHint = useMemo(() => {
    const parse = parseDateTokens(gallery.searchTerm);
    const trailing = matchTrailingDateToken(gallery.searchTerm);
    const invalid = parse.invalidTokens.find(
      (token) =>
        !(
          trailing &&
          token.key === trailing.key &&
          token.raw === trailing.partialValue &&
          isPossibleDatePrefix(token.raw)
        )
    );

    if (invalid) {
      return { isInvalid: true, label: t('widgets.gallery.dateFilterInvalid', { value: invalid.raw }) };
    }

    if (!parse.range) {
      return null;
    }

    const locale = i18n.language;
    const { from, to } = parse.range;

    if (from !== undefined && to !== undefined) {
      return {
        isInvalid: false,
        label:
          from === to
            ? t('widgets.gallery.dateFilterDay', { date: formatIsoDate(from, locale) })
            : t('widgets.gallery.dateFilterRange', {
                from: formatIsoDate(from, locale),
                to: formatIsoDate(to, locale),
              }),
      };
    }

    if (from !== undefined) {
      return { isInvalid: false, label: t('widgets.gallery.dateFilterFrom', { date: formatIsoDate(from, locale) }) };
    }

    return {
      isInvalid: false,
      label: t('widgets.gallery.dateFilterThrough', { date: formatIsoDate(to ?? '', locale) }),
    };
  }, [gallery.searchTerm, i18n.language, t]);

  const searchInput = useMemo(
    () => (
      <Stack gap="1">
        <InputGroup startElement={SEARCH_START_ELEMENT} endElement={searchClearButton}>
          <Input
            aria-describedby={dateHint?.isInvalid ? SEARCH_DATE_HINT_ID : undefined}
            aria-invalid={dateHint?.isInvalid || undefined}
            aria-label={t('widgets.gallery.searchImagesAriaLabel')}
            placeholder={t('widgets.gallery.searchImagesPlaceholder')}
            size="xs"
            value={gallery.searchTerm}
            onChange={handleSearchChange}
          />
        </InputGroup>
        {dateHint ? (
          <Text
            color={dateHint.isInvalid ? 'fg.error' : 'fg.subtle'}
            fontSize="2xs"
            id={SEARCH_DATE_HINT_ID}
            role="status"
          >
            {dateHint.label}
          </Text>
        ) : null}
      </Stack>
    ),
    [dateHint, gallery.searchTerm, handleSearchChange, searchClearButton, t]
  );

  return (
    <Stack gap="2">
      {isWide ? (
        <HStack gap="2">
          <Box flex="1" maxW="16rem" minW="10rem">
            <GalleryBoardSelect />
          </Box>
          {viewTabs}
          <Spacer />
          <Box flex="1" maxW="20rem" minW="12rem">
            {searchInput}
          </Box>
          {toolbarActions}
        </HStack>
      ) : (
        <>
          <GalleryBoardSelect />
          <HStack justify="space-between">
            {viewTabs}
            {toolbarActions}
          </HStack>
        </>
      )}
      {!isWide && searchInput}
    </Stack>
  );
};

const GalleryUploadButton = () => {
  const { t } = useTranslation();
  const { actions, gallery } = useGalleryWidget();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const selectedBoard = gallery.boards.find((board) => board.id === gallery.selectedBoardId);
  const isVirtualTarget = isDateBoardId(gallery.selectedBoardId);

  const handleFileChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(event.currentTarget.files ?? []);

      event.currentTarget.value = '';

      if (files.length > 0) {
        void actions.uploadFiles(files);
      }
    },
    [actions]
  );

  const handleUploadClick = useCallback(() => fileInputRef.current?.click(), []);

  return (
    <>
      <input
        accept={ACCEPTED_UPLOAD_EXTENSIONS}
        multiple
        ref={fileInputRef}
        style={UPLOAD_INPUT_STYLE}
        type="file"
        onChange={handleFileChange}
      />
      <IconButton
        aria-label={
          isVirtualTarget
            ? t('widgets.gallery.uploadsUnavailableForDateBoards')
            : t('widgets.gallery.uploadImagesToBoard', {
                name: selectedBoard?.name ?? t('widgets.gallery.selectedBoardFallback'),
              })
        }
        disabled={isVirtualTarget}
        size="xs"
        variant="outline"
        onClick={handleUploadClick}
      >
        <UploadIcon />
      </IconButton>
    </>
  );
};

const GallerySettingsMenu = () => {
  const { t } = useTranslation();
  const { actions, gallery } = useGalleryWidget();
  const {
    imageDensityPercent,
    imageOrderDir,
    paginationMode,
    showImageDimensions,
    showPendingItems,
    starredFirst,
    thumbnailFit,
  } = gallery.settings;
  const handleDensityKeyDown = useCallback((event: React.KeyboardEvent) => event.stopPropagation(), []);
  const handleDensityChange = useCallback(
    (event: { value: number[] }) =>
      actions.updateSettings({ imageDensityPercent: event.value[0] ?? imageDensityPercent }),
    [actions, imageDensityPercent]
  );
  const handleSquareFit = useCallback(() => actions.updateSettings({ thumbnailFit: 'square' }), [actions]);
  const handleAspectFit = useCallback(() => actions.updateSettings({ thumbnailFit: 'aspect' }), [actions]);
  const handleShowDimensionsChange = useCallback(
    (showImageDimensions: boolean) => actions.updateSettings({ showImageDimensions }),
    [actions]
  );
  const handleShowPendingItemsChange = useCallback(
    (showPendingItems: boolean) => actions.updateSettings({ showPendingItems }),
    [actions]
  );
  const handleNewestFirst = useCallback(() => actions.updateSettings({ imageOrderDir: 'DESC' }), [actions]);
  const handleOldestFirst = useCallback(() => actions.updateSettings({ imageOrderDir: 'ASC' }), [actions]);
  const handleStarredFirstChange = useCallback(
    (starredFirst: boolean) => actions.updateSettings({ starredFirst }),
    [actions]
  );
  const handleInfinitePagination = useCallback(() => actions.updateSettings({ paginationMode: 'infinite' }), [actions]);
  const handlePaginatedPagination = useCallback(
    () => actions.updateSettings({ paginationMode: 'paginated' }),
    [actions]
  );
  const densityAriaLabel = useMemo(() => [t('widgets.gallery.imageDensity')], [t]);
  const densityValue = useMemo(() => [imageDensityPercent], [imageDensityPercent]);

  return (
    <Menu.Root positioning={GALLERY_SETTINGS_POSITIONING}>
      <Menu.Trigger asChild>
        <IconButton aria-label={t('widgets.gallery.settings')} size="xs" variant="outline">
          <SettingsIcon />
        </IconButton>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <Menu.Content minW="14rem" p="3">
            <Stack gap="3">
              <Stack gap="2">
                <HStack justify="space-between">
                  <SettingLabel>{t('widgets.gallery.settingsGridDensity')}</SettingLabel>
                  <Text color="fg.subtle" fontSize="2xs">
                    {imageDensityPercent}%
                  </Text>
                </HStack>
                <Slider.Root
                  aria-label={densityAriaLabel}
                  max={100}
                  min={0}
                  step={1}
                  value={densityValue}
                  onKeyDown={handleDensityKeyDown}
                  onValueChange={handleDensityChange}
                >
                  <Slider.Control>
                    <Slider.Track>
                      <Slider.Range />
                    </Slider.Track>
                    <Slider.Thumbs />
                  </Slider.Control>
                </Slider.Root>
              </Stack>
              <Stack gap="2">
                <SettingLabel>{t('widgets.gallery.settingsThumbnails')}</SettingLabel>
                <HStack gap="1">
                  <Button
                    flex="1"
                    size="2xs"
                    variant={thumbnailFit === 'square' ? 'solid' : 'outline'}
                    onClick={handleSquareFit}
                  >
                    {t('widgets.gallery.thumbnailFitSquare')}
                  </Button>
                  <Button
                    flex="1"
                    size="2xs"
                    variant={thumbnailFit === 'aspect' ? 'solid' : 'outline'}
                    onClick={handleAspectFit}
                  >
                    {t('widgets.gallery.thumbnailFitAspect')}
                  </Button>
                </HStack>
                <SettingSwitch
                  checked={showImageDimensions}
                  label={t('widgets.gallery.alwaysShowDimensions')}
                  onChange={handleShowDimensionsChange}
                />
                <SettingSwitch
                  checked={showPendingItems}
                  label={t('widgets.gallery.showPendingItems')}
                  onChange={handleShowPendingItemsChange}
                />
              </Stack>
              <Stack gap="2">
                <SettingLabel>{t('widgets.gallery.imageSort')}</SettingLabel>
                <HStack gap="1">
                  <Button
                    flex="1"
                    size="2xs"
                    variant={imageOrderDir === 'DESC' ? 'solid' : 'outline'}
                    onClick={handleNewestFirst}
                  >
                    {t('widgets.gallery.newest')}
                  </Button>
                  <Button
                    flex="1"
                    size="2xs"
                    variant={imageOrderDir === 'ASC' ? 'solid' : 'outline'}
                    onClick={handleOldestFirst}
                  >
                    {t('widgets.gallery.oldest')}
                  </Button>
                </HStack>
                <SettingSwitch
                  checked={starredFirst}
                  label={t('widgets.gallery.showStarredImagesFirst')}
                  onChange={handleStarredFirstChange}
                />
              </Stack>
              <Stack gap="2">
                <SettingLabel>{t('widgets.gallery.pagination')}</SettingLabel>
                <HStack gap="1">
                  <Button
                    flex="1"
                    size="2xs"
                    variant={paginationMode === 'infinite' ? 'solid' : 'outline'}
                    onClick={handleInfinitePagination}
                  >
                    {t('widgets.gallery.infinite')}
                  </Button>
                  <Button
                    flex="1"
                    size="2xs"
                    variant={paginationMode === 'paginated' ? 'solid' : 'outline'}
                    onClick={handlePaginatedPagination}
                  >
                    {t('common.pages')}
                  </Button>
                </HStack>
              </Stack>
            </Stack>
          </Menu.Content>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const SettingSwitch = ({
  checked,
  label,
  onChange,
}: {
  checked: boolean;
  label: string;
  onChange: (checked: boolean) => void;
}) => {
  const handleCheckedChange = useCallback((event: { checked: boolean }) => onChange(event.checked), [onChange]);

  return (
    <Switch.Root checked={checked} size="sm" onCheckedChange={handleCheckedChange}>
      <Switch.HiddenInput />
      <Switch.Control>
        <Switch.Thumb />
      </Switch.Control>
      <Switch.Label fontSize="2xs">{label}</Switch.Label>
    </Switch.Root>
  );
};

const SettingLabel = ({ children }: { children: string }) => (
  <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
    {children}
  </Text>
);
