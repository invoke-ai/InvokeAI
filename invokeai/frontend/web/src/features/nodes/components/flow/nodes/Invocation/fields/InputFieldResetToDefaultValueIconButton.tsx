import { IconButton } from '@invoke-ai/ui-library';
import { useInputFieldDefaultValue } from 'features/nodes/hooks/useInputFieldDefaultValue';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

type Props = {
  nodeId: string;
  fieldName: string;
};

export const InputFieldResetToDefaultValueIconButton = memo(({ nodeId, fieldName }: Props) => {
  const { t } = useTranslation();
  const { isValueChanged, resetToDefaultValue } = useInputFieldDefaultValue(nodeId, fieldName);

  return (
    <IconButton
      variant="ghost"
      tooltip={t('nodes.resetToDefaultValue')}
      aria-label={t('nodes.resetToDefaultValue')}
      icon={<PiArrowCounterClockwiseBold />}
      pointerEvents="auto"
      size="xs"
      onClick={resetToDefaultValue}
      isDisabled={!isValueChanged}
    />
  );
});

InputFieldResetToDefaultValueIconButton.displayName = 'InputFieldResetToDefaultValueIconButton';
