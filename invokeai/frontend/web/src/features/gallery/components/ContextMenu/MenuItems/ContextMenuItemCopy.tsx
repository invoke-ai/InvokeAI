import { IconMenuItem } from 'common/components/IconMenuItem';
import { useCopyImageToClipboard } from 'common/hooks/useCopyImageToClipboard';
import { useItemDTOContext } from 'features/gallery/contexts/ItemDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold } from 'react-icons/pi';
import { isImageDTO } from 'services/api/types';

export const ContextMenuItemCopy = memo(() => {
  const { t } = useTranslation();
  const itemDTO = useItemDTOContext();
  const copyImageToClipboard = useCopyImageToClipboard();

  const onClick = useCallback(() => {
    if (isImageDTO(itemDTO)) {
      copyImageToClipboard(itemDTO.image_url);
    } else {
      // copyVideoToClipboard(itemDTO.video_url);
    }
  }, [copyImageToClipboard, itemDTO]);

  return (
    <IconMenuItem
      icon={<PiCopyBold />}
      aria-label={t('parameters.copyImage')}
      tooltip={t('parameters.copyImage')}
      onClickCapture={onClick}
    />
  );
});

ContextMenuItemCopy.displayName = 'ContextMenuItemCopy';
