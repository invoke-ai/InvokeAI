import { Box, HStack, Input, Menu, Portal, Slider, Spacer, Stack, Switch, Text } from '@chakra-ui/react';
import { useRef } from 'react';
import { PiGearSixBold, PiUploadSimpleBold } from 'react-icons/pi';

import { Button, IconButton } from '../../components/ui/Button';
import { isDateBoardId } from '../../gallery/api';
import { GalleryBoardSelect } from './GalleryBoardSelect';
import { useGalleryWidget } from './GalleryWidgetContext';

const ACCEPTED_UPLOAD_EXTENSIONS = 'image/png,image/jpeg,image/webp';

export const GalleryToolbar = ({ layout }: { layout: 'stacked' | 'wide' }) => {
  const { actions, gallery } = useGalleryWidget();
  const isWide = layout === 'wide';
  const viewTabs = (
    <HStack gap="1">
      <Button
        size="xs"
        variant={gallery.galleryView === 'images' ? 'solid' : 'outline'}
        onClick={() => actions.setView('images')}
      >
        Images
      </Button>
      <Button
        size="xs"
        variant={gallery.galleryView === 'assets' ? 'solid' : 'outline'}
        onClick={() => actions.setView('assets')}
      >
        Assets
      </Button>
    </HStack>
  );
  const toolbarActions = (
    <HStack gap="2">
      <Text color="fg.subtle" fontSize="2xs">
        {gallery.images.length} loaded
      </Text>
      <GalleryUploadButton />
      <GallerySettingsMenu />
    </HStack>
  );

  return (
    <Stack gap="2">
      {isWide ? (
        <HStack gap="2">
          <Box flex="1" maxW="24rem" minW="10rem">
            <GalleryBoardSelect />
          </Box>
          {viewTabs}
          <Spacer />
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
      <Input
        aria-label="Search gallery images"
        placeholder="Search images"
        size="sm"
        value={gallery.searchTerm}
        onChange={(event) => actions.setSearchTerm(event.currentTarget.value)}
      />
    </Stack>
  );
};

const GalleryUploadButton = () => {
  const { actions, gallery } = useGalleryWidget();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const selectedBoard = gallery.boards.find((board) => board.id === gallery.selectedBoardId);
  const isVirtualTarget = isDateBoardId(gallery.selectedBoardId);

  return (
    <>
      <input
        accept={ACCEPTED_UPLOAD_EXTENSIONS}
        multiple
        ref={fileInputRef}
        style={{ display: 'none' }}
        type="file"
        onChange={(event) => {
          const files = Array.from(event.currentTarget.files ?? []);

          event.currentTarget.value = '';

          if (files.length > 0) {
            void actions.uploadFiles(files);
          }
        }}
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
        onClick={() => fileInputRef.current?.click()}
      >
        <PiUploadSimpleBold />
      </IconButton>
    </>
  );
};

const GallerySettingsMenu = () => {
  const { actions, gallery } = useGalleryWidget();
  const { imageDensityPercent, imageOrderDir, paginationMode, showImageDimensions, starredFirst, thumbnailFit } =
    gallery.settings;

  return (
    <Menu.Root positioning={{ placement: 'bottom-end' }}>
      <Menu.Trigger asChild>
        <IconButton aria-label="Gallery settings" size="xs" variant="outline">
          <PiGearSixBold />
        </IconButton>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <Menu.Content
            bg="bg.surfaceRaised"
            borderWidth="1px"
            borderColor="border.emphasis"
            color="fg.default"
            minW="14rem"
            p="3"
            rounded="lg"
            shadow="lg"
          >
            <Stack gap="3">
              <Stack gap="2">
                <HStack justify="space-between">
                  <SettingLabel>Grid Density</SettingLabel>
                  <Text color="fg.subtle" fontSize="2xs">
                    {imageDensityPercent}%
                  </Text>
                </HStack>
                <Slider.Root
                  aria-label={['Gallery image density']}
                  max={100}
                  min={0}
                  step={1}
                  value={[imageDensityPercent]}
                  onKeyDown={(event) => event.stopPropagation()}
                  onValueChange={(event) =>
                    actions.updateSettings({ imageDensityPercent: event.value[0] ?? imageDensityPercent })
                  }
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
                    onClick={() => actions.updateSettings({ thumbnailFit: 'square' })}
                  >
                    Square
                  </Button>
                  <Button
                    flex="1"
                    size="2xs"
                    variant={thumbnailFit === 'aspect' ? 'solid' : 'outline'}
                    onClick={() => actions.updateSettings({ thumbnailFit: 'aspect' })}
                  >
                    Aspect
                  </Button>
                </HStack>
                <SettingSwitch
                  checked={showImageDimensions}
                  label="Always show dimensions"
                  onChange={(checked) => actions.updateSettings({ showImageDimensions: checked })}
                />
              </Stack>
              <Stack gap="2">
                <SettingLabel>Image Sort</SettingLabel>
                <HStack gap="1">
                  <Button
                    flex="1"
                    size="2xs"
                    variant={imageOrderDir === 'DESC' ? 'solid' : 'outline'}
                    onClick={() => actions.updateSettings({ imageOrderDir: 'DESC' })}
                  >
                    Newest
                  </Button>
                  <Button
                    flex="1"
                    size="2xs"
                    variant={imageOrderDir === 'ASC' ? 'solid' : 'outline'}
                    onClick={() => actions.updateSettings({ imageOrderDir: 'ASC' })}
                  >
                    Oldest
                  </Button>
                </HStack>
                <SettingSwitch
                  checked={starredFirst}
                  label="Show starred images first"
                  onChange={(checked) => actions.updateSettings({ starredFirst: checked })}
                />
              </Stack>
              <Stack gap="2">
                <SettingLabel>Pagination</SettingLabel>
                <HStack gap="1">
                  <Button
                    flex="1"
                    size="2xs"
                    variant={paginationMode === 'infinite' ? 'solid' : 'outline'}
                    onClick={() => actions.updateSettings({ paginationMode: 'infinite' })}
                  >
                    Infinite
                  </Button>
                  <Button
                    flex="1"
                    size="2xs"
                    variant={paginationMode === 'paginated' ? 'solid' : 'outline'}
                    onClick={() => actions.updateSettings({ paginationMode: 'paginated' })}
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
}) => (
  <Switch.Root checked={checked} size="sm" onCheckedChange={(event) => onChange(event.checked)}>
    <Switch.HiddenInput />
    <Switch.Control>
      <Switch.Thumb />
    </Switch.Control>
    <Switch.Label fontSize="2xs">{label}</Switch.Label>
  </Switch.Root>
);

const SettingLabel = ({ children }: { children: string }) => (
  <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
    {children}
  </Text>
);
