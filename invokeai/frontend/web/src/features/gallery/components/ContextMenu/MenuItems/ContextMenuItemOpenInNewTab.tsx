import { useAppDispatch } from 'app/store/storeHooks';
import { IconMenuItem } from 'common/components/IconMenuItem';
import { useItemDTOContext } from 'features/gallery/contexts/ItemDTOContext';
import { imageOpenedInNewTab } from 'features/gallery/store/actions';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowSquareOutBold } from 'react-icons/pi';
import { isImageDTO } from 'services/api/types';

export const ContextMenuItemOpenInNewTab = memo(() => {
  const { t } = useTranslation();
  const itemDTO = useItemDTOContext();
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    if (isImageDTO(itemDTO)) {
    window.open(itemDTO.image_url, '_blank');
    dispatch(imageOpenedInNewTab());
    } else {
      // TODO: Implement video open in new tab
    }
  }, [itemDTO, dispatch]);

  return (
    <IconMenuItem
      onClickCapture={onClick}
      aria-label={t('common.openInNewTab')}
      tooltip={t('common.openInNewTab')}
      icon={<PiArrowSquareOutBold />}
    />
  );
});

ContextMenuItemOpenInNewTab.displayName = 'ContextMenuItemOpenInNewTab';
