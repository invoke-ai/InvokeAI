import { useAppSelector } from 'app/store/storeHooks';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import type { InvIconButtonProps } from 'common/components/InvIconButton/types';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

type DeleteImageButtonProps = Omit<InvIconButtonProps, 'aria-label'> & {
  onClick: () => void;
};

export const DeleteImageButton = memo((props: DeleteImageButtonProps) => {
  const { onClick, isDisabled } = props;
  const { t } = useTranslation();
  const isConnected = useAppSelector((s) => s.system.isConnected);

  return (
    <InvIconButton
      onClick={onClick}
      icon={<PiTrashSimpleBold />}
      tooltip={`${t('gallery.deleteImage')} (Del)`}
      aria-label={`${t('gallery.deleteImage')} (Del)`}
      isDisabled={isDisabled || !isConnected}
      colorScheme="error"
    />
  );
});

DeleteImageButton.displayName = 'DeleteImageButton';
