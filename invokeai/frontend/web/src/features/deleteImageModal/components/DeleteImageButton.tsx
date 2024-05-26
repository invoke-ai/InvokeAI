import type { IconButtonProps } from '@invoke-ai/ui-library';
import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

type DeleteImageButtonProps = Omit<IconButtonProps, 'aria-label'> & {
  onClick: () => void;
};

export const DeleteImageButton = memo((props: DeleteImageButtonProps) => {
  const { onClick, isDisabled } = props;
  const { t } = useTranslation();
  const isConnected = useAppSelector((s) => s.system.isConnected);
  const imageSelectionLength: number = useAppSelector((s) => s.gallery.selection.length);
  const labelMessage: string = `${t('gallery.deleteImage', { count: imageSelectionLength })} (Del)`;

  return (
    <IconButton
      onClick={onClick}
      icon={<PiTrashSimpleBold />}
      tooltip={labelMessage}
      aria-label={labelMessage}
      isDisabled={isDisabled || !isConnected}
      colorScheme="error"
    />
  );
});

DeleteImageButton.displayName = 'DeleteImageButton';
