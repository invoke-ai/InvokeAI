import { IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import SingleSelectionMenuItems from 'features/gallery/components/ImageContextMenu/SingleSelectionMenuItems';
import { useCurrentImageDTO } from 'features/viewer/hooks/useCurrentImageDTO';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDotsThreeOutlineFill } from 'react-icons/pi';

export const ViewerToolbarImageMenu = memo(() => {
  const imageDTO = useCurrentImageDTO();
  const { t } = useTranslation();

  return (
    <Menu isLazy>
      <MenuButton
        as={IconButton}
        aria-label={t('parameters.imageActions')}
        tooltip={t('parameters.imageActions')}
        isDisabled={!imageDTO}
        icon={<PiDotsThreeOutlineFill />}
      />
      <MenuList>{imageDTO && <SingleSelectionMenuItems imageDTO={imageDTO} />}</MenuList>
    </Menu>
  );
});

ViewerToolbarImageMenu.displayName = 'ViewerToolbarImageMenu';
