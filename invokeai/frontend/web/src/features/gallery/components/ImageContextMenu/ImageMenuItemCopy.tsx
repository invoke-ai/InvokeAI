import { IconButton } from '@invoke-ai/ui-library';
import { useCopyImageToClipboard } from 'common/hooks/useCopyImageToClipboard';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold } from 'react-icons/pi';

export const ImageMenuItemCopy = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();
  const { isClipboardAPIAvailable, copyImageToClipboard } = useCopyImageToClipboard();

  const onPointerUp = useCallback(() => {
    copyImageToClipboard(imageDTO.image_url);
  }, [copyImageToClipboard, imageDTO.image_url]);

  if (!isClipboardAPIAvailable) {
    return null;
  }

  return (
    <IconButton
      icon={<PiCopyBold />}
      aria-label={t('parameters.copyImage')}
      tooltip={t('parameters.copyImage')}
      onPointerUpCapture={onPointerUp}
      variant="ghost"
      colorScheme="base"
    />
  );
});

ImageMenuItemCopy.displayName = 'ImageMenuItemCopy';
