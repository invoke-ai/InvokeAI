import { IconButton } from '@invoke-ai/ui-library';
import { stopPropagation } from 'common/util/stopPropagation';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCheckBold } from 'react-icons/pi';

type Props = {
  isEnabled: boolean;
  onToggle: () => void;
};

export const CanvasEntityEnabledToggle = memo(({ isEnabled, onToggle }: Props) => {
  const { t } = useTranslation();

  return (
    <IconButton
      size="sm"
      aria-label={t(isEnabled ? 'common.enabled' : 'common.disabled')}
      tooltip={t(isEnabled ? 'common.enabled' : 'common.disabled')}
      variant="outline"
      icon={isEnabled ? <PiCheckBold /> : undefined}
      onClick={onToggle}
      colorScheme="base"
      onDoubleClick={stopPropagation} // double click expands the layer
    />
  );
});

CanvasEntityEnabledToggle.displayName = 'CanvasEntityEnabledToggle';
