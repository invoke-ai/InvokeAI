import {
  Divider,
  Flex,
  IconButton,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Portal,
  Text,
} from '@invoke-ai/ui-library';
import AlwaysShowImageSizeCheckbox from 'features/gallery/components/GallerySettingsPopover/AlwaysShowImageSizeCheckbox';
import AutoSwitchCheckbox from 'features/gallery/components/GallerySettingsPopover/AutoSwitchCheckbox';
import ImageMinimumWidthSlider from 'features/gallery/components/GallerySettingsPopover/ImageMinimumWidthSlider';
import ShowStarredFirstCheckbox from 'features/gallery/components/GallerySettingsPopover/ShowStarredFirstCheckbox';
import SortDirectionCombobox from 'features/gallery/components/GallerySettingsPopover/SortDirectionCombobox';
import UsePagedGalleryViewCheckbox from 'features/gallery/components/GallerySettingsPopover/UsePagedGalleryViewCheckbox';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiGearSixFill } from 'react-icons/pi';

export const GallerySettingsPopover = memo(() => {
  const { t } = useTranslation();

  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton
          size="sm"
          variant="link"
          alignSelf="stretch"
          aria-label={t('gallery.imagesSettings')}
          icon={<PiGearSixFill />}
          tooltip={t('gallery.imagesSettings')}
        />
      </PopoverTrigger>
      <Portal>
        <PopoverContent>
          <PopoverArrow />
          <PopoverBody>
            <Flex direction="column" gap={2}>
              <Text fontWeight="semibold" color="base.300">
                {t('gallery.gallerySettings')}
              </Text>

              <Divider />

              <ImageMinimumWidthSlider />
              <UsePagedGalleryViewCheckbox />
              <AutoSwitchCheckbox />
              <AlwaysShowImageSizeCheckbox />

              <Divider />

              <ShowStarredFirstCheckbox />
              <SortDirectionCombobox />
            </Flex>
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
});
GallerySettingsPopover.displayName = 'GallerySettingsPopover';
