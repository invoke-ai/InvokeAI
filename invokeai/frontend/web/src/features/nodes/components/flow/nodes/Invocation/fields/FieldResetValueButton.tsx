import type { IconButtonProps } from '@invoke-ai/ui-library';
import { IconButton } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

type Props = Omit<IconButtonProps, 'aria-label'>;

export const FieldResetValueButton = memo((props: Props) => {
  const { t } = useTranslation();

  return (
    <IconButton
      variant="ghost"
      tooltip={t('nodes.resetToDefaultValue')}
      aria-label={t('nodes.resetToDefaultValue')}
      icon={<PiArrowCounterClockwiseBold />}
      pointerEvents="auto"
      size="xs"
      {...props}
    />
  );
});

FieldResetValueButton.displayName = 'FieldResetValueButton';
