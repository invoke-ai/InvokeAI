import { IconButton } from '@invoke-ai/ui-library';
import { useInputFieldInitialLinearViewValue } from 'features/nodes/hooks/useInputFieldInitialLinearViewValue';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

type Props = {
  nodeId: string;
  fieldName: string;
};

export const InputFieldResetToInitialValueIconButton = memo(({ nodeId, fieldName }: Props) => {
  const { t } = useTranslation();
  const { isValueChanged, resetToInitialLinearViewValue } = useInputFieldInitialLinearViewValue(nodeId, fieldName);

  return (
    <IconButton
      variant="ghost"
      tooltip={t('nodes.resetToDefaultValue')}
      aria-label={t('nodes.resetToDefaultValue')}
      icon={<PiArrowCounterClockwiseBold />}
      pointerEvents="auto"
      size="xs"
      onClick={resetToInitialLinearViewValue}
      isDisabled={!isValueChanged}
    />
  );
});

InputFieldResetToInitialValueIconButton.displayName = 'InputFieldResetToInitialValueIconButton';
