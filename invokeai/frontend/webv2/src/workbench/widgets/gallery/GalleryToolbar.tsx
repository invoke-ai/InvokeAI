import type { GalleryView } from '@workbench/gallery/api';

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
import { Button, CloseButton, IconButton, Tabs } from '@workbench/components/ui';
import { isDateBoardId } from '@workbench/gallery/api';
import { SearchIcon, SettingsIcon, UploadIcon } from 'lucide-react';
import { useCallback, useMemo, useRef } from 'react';

import { GalleryBoardSelect } from './GalleryBoardSelect';
import { useGalleryWidget } from './GalleryWidgetContext';

const ACCEPTED_UPLOAD_EXTENSIONS = 'image/png,image/jpeg,image/webp';
const galleryViewTabs = [
  { label: 'Images', value: 'images' },
  { label: 'Assets', value: 'assets' },
] satisfies { label: string; value: GalleryView }[];

const UPLOAD_INPUT_STYLE = { display: 'none' } as const;
const GALLERY_SETTINGS_POSITIONING = { placement: 'bottom-end' } as const;
const SLIDER_ARIA_LABEL = ['Gallery image density'];
const SEARCH_START_ELEMENT = <Icon as={SearchIcon} size="sm" />;

export const GalleryToolbar = ({ layout }: { layout: 'stacked' | 'wide' }) => {
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
        <CloseButton size="2xs" aria-label="Clear search" onClick={handleClearSearch} me="-2" />
      ) : null,
    [gallery.searchTerm, handleClearSearch]
  );

  const viewTabs = useMemo(
    () => (
      <Tabs.Root size="sm" variant="subtle" value={gallery.galleryView} onValueChange={handleViewChange}>
        <Tabs.List>
          {galleryViewTabs.map((item) => (
            <Tabs.Trigger key={item.value} value={item.value} fontSize="xs">
              {item.label}
            </Tabs.Trigger>
          ))}
        </Tabs.List>
      </Tabs.Root>
    ),
    [gallery.galleryView, handleViewChange]
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

  const searchInput = useMemo(
    () => (
      <InputGroup startElement={SEARCH_START_ELEMENT} endElement={searchClearButton}>
        <Input
          aria-label="Search gallery images"
          placeholder="Search images"
          size="xs"
          value={gallery.searchTerm}
          onChange={handleSearchChange}
        />
      </InputGroup>
    ),
    [gallery.searchTerm, handleSearchChange, searchClearButton]
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
            ? 'Uploads are unavailable for date boards'
            : `Upload images to ${selectedBoard?.name ?? 'the selected board'}`
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
  const { actions, gallery } = useGalleryWidget();
  const { imageDensityPercent, imageOrderDir, paginationMode, showImageDimensions, starredFirst, thumbnailFit } =
    gallery.settings;
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
  const densityValue = useMemo(() => [imageDensityPercent], [imageDensityPercent]);

  return (
    <Menu.Root positioning={GALLERY_SETTINGS_POSITIONING}>
      <Menu.Trigger asChild>
        <IconButton aria-label="Gallery settings" size="xs" variant="outline">
          <SettingsIcon />
        </IconButton>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <Menu.Content minW="14rem" p="3">
            <Stack gap="3">
              <Stack gap="2">
                <HStack justify="space-between">
                  <SettingLabel>Grid Density</SettingLabel>
                  <Text color="fg.subtle" fontSize="2xs">
                    {imageDensityPercent}%
                  </Text>
                </HStack>
                <Slider.Root
                  aria-label={SLIDER_ARIA_LABEL}
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
                <SettingLabel>Thumbnails</SettingLabel>
                <HStack gap="1">
                  <Button
                    flex="1"
                    size="2xs"
                    variant={thumbnailFit === 'square' ? 'solid' : 'outline'}
                    onClick={handleSquareFit}
                  >
                    Square
                  </Button>
                  <Button
                    flex="1"
                    size="2xs"
                    variant={thumbnailFit === 'aspect' ? 'solid' : 'outline'}
                    onClick={handleAspectFit}
                  >
                    Aspect
                  </Button>
                </HStack>
                <SettingSwitch
                  checked={showImageDimensions}
                  label="Always show dimensions"
                  onChange={handleShowDimensionsChange}
                />
              </Stack>
              <Stack gap="2">
                <SettingLabel>Image Sort</SettingLabel>
                <HStack gap="1">
                  <Button
                    flex="1"
                    size="2xs"
                    variant={imageOrderDir === 'DESC' ? 'solid' : 'outline'}
                    onClick={handleNewestFirst}
                  >
                    Newest
                  </Button>
                  <Button
                    flex="1"
                    size="2xs"
                    variant={imageOrderDir === 'ASC' ? 'solid' : 'outline'}
                    onClick={handleOldestFirst}
                  >
                    Oldest
                  </Button>
                </HStack>
                <SettingSwitch
                  checked={starredFirst}
                  label="Show starred images first"
                  onChange={handleStarredFirstChange}
                />
              </Stack>
              <Stack gap="2">
                <SettingLabel>Pagination</SettingLabel>
                <HStack gap="1">
                  <Button
                    flex="1"
                    size="2xs"
                    variant={paginationMode === 'infinite' ? 'solid' : 'outline'}
                    onClick={handleInfinitePagination}
                  >
                    Infinite
                  </Button>
                  <Button
                    flex="1"
                    size="2xs"
                    variant={paginationMode === 'paginated' ? 'solid' : 'outline'}
                    onClick={handlePaginatedPagination}
                  >
                    Pages
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
