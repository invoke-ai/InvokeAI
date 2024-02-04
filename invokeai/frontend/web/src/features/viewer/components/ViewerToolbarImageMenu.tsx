import { IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import SingleSelectionMenuItems from 'features/gallery/components/ImageContextMenu/SingleSelectionMenuItems';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { selectIsDisabledToolbarImageButtons } from 'features/viewer/store/viewerSelectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDotsThreeOutlineFill } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

export const ViewerToolbarImageMenu = memo(() => {
  const lastSelectedImage = useAppSelector(selectLastSelectedImage);
  const isDisabled = useAppSelector(selectIsDisabledToolbarImageButtons);
  const { t } = useTranslation();

  const { currentData: imageDTO } = useGetImageDTOQuery(lastSelectedImage?.image_name ?? skipToken);

  return (
    <Menu isLazy>
      <MenuButton
        as={IconButton}
        aria-label={t('parameters.imageActions')}
        tooltip={t('parameters.imageActions')}
        isDisabled={!imageDTO || isDisabled}
        icon={<PiDotsThreeOutlineFill />}
      />
      <MenuList>{imageDTO && <SingleSelectionMenuItems imageDTO={imageDTO} />}</MenuList>
    </Menu>
  );
});

ViewerToolbarImageMenu.displayName = 'ViewerToolbarImageMenu';
