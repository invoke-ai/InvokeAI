import { Divider, Flex, IconButton, Popover, PopoverBody, PopoverContent, PopoverTrigger } from '@invoke-ai/ui-library';
import AlwaysShowImageSizeCheckbox from 'features/gallery/components/GallerySettingsPopover/AlwaysShowImageSizeCheckbox';
import AutoSwitchCheckbox from 'features/gallery/components/GallerySettingsPopover/AutoSwitchCheckbox';
import ImageMinimumWidthSlider from 'features/gallery/components/GallerySettingsPopover/ImageMinimumWidthSlider';
import ShowStarredFirstCheckbox from 'features/gallery/components/GallerySettingsPopover/ShowStarredFirstCheckbox';
import SortDirectionCombobox from 'features/gallery/components/GallerySettingsPopover/SortDirectionCombobox';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiGearSixFill } from 'react-icons/pi';

const GallerySettingsPopover = () => {
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
      <PopoverContent>
        <PopoverBody>
          <Flex direction="column" gap={2}>
            <ImageMinimumWidthSlider />
            <AutoSwitchCheckbox />
            <AlwaysShowImageSizeCheckbox />
            <Divider pt={2} />
            <ShowStarredFirstCheckbox />
            <SortDirectionCombobox />
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

export default memo(GallerySettingsPopover);
