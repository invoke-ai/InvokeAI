import type { IconButtonProps } from '@invoke-ai/ui-library';
import { IconButton } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

type Props = Omit<IconButtonProps, 'aria-label'> & {
  onDelete: () => void;
};

export const InpaintMaskDeleteModifierButton = memo(({ onDelete, ...rest }: Props) => {
  const { t } = useTranslation();
  return (
    <IconButton
      tooltip={t('common.delete')}
      variant="link"
      aria-label={t('common.delete')}
      icon={<PiXBold />}
      onClick={onDelete}
      flexGrow={0}
      size="sm"
      p={0}
      colorScheme="error"
      {...rest}
    />
  );
});

InpaintMaskDeleteModifierButton.displayName = 'InpaintMaskDeleteNoiseButton';
