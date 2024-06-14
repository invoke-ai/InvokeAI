import { IconButton } from '@invoke-ai/ui-library';
import { stopPropagation } from 'common/util/stopPropagation';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

type Props = { onDelete: () => void };

export const EntityDeleteButton = memo(({ onDelete }: Props) => {
  const { t } = useTranslation();
  return (
    <IconButton
      size="sm"
      colorScheme="error"
      aria-label={t('common.delete')}
      tooltip={t('common.delete')}
      icon={<PiTrashSimpleBold />}
      onClick={onDelete}
      onDoubleClick={stopPropagation} // double click expands the layer
    />
  );
});

EntityDeleteButton.displayName = 'EntityDeleteButton';
