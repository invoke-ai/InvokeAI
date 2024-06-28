import { Divider, Flex, IconButton, Popover, PopoverBody, PopoverContent, PopoverTrigger } from '@invoke-ai/ui-library';
import BoardAutoAddSelect from 'features/gallery/components/Boards/BoardAutoAddSelect';
import AlwaysShowImageSizeCheckbox from 'features/gallery/components/GallerySettingsPopover/AlwaysShowImageSizeCheckbox';
import AutoAssignBoardCheckbox from 'features/gallery/components/GallerySettingsPopover/AutoAssignBoardCheckbox';
import AutoSwitchCheckbox from 'features/gallery/components/GallerySettingsPopover/AutoSwitchCheckbox';
import ImageMinimumWidthSlider from 'features/gallery/components/GallerySettingsPopover/ImageMinimumWidthSlider';
import ShowArchivedBoardsCheckbox from 'features/gallery/components/GallerySettingsPopover/ShowArchivedBoardsCheckbox';
import ShowStarredFirstCheckbox from 'features/gallery/components/GallerySettingsPopover/ShowStarredFirstCheckbox';
import SortDirectionCombobox from 'features/gallery/components/GallerySettingsPopover/SortDirectionCombobox';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { RiSettings4Fill } from 'react-icons/ri';

const GallerySettingsPopover = () => {
  const { t } = useTranslation();

  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton aria-label={t('gallery.gallerySettings')} size="sm" icon={<RiSettings4Fill />} />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody>
          <Flex direction="column" gap={2}>
            <ImageMinimumWidthSlider />
            <AutoSwitchCheckbox />
            <AutoAssignBoardCheckbox />
            <AlwaysShowImageSizeCheckbox />
            <ShowArchivedBoardsCheckbox />
            <BoardAutoAddSelect />
            <Divider />
            <ShowStarredFirstCheckbox />
            <SortDirectionCombobox />
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

export default memo(GallerySettingsPopover);
