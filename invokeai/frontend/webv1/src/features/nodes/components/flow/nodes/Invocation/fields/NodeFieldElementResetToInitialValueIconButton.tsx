import { IconButton } from '@invoke-ai/ui-library';
import { useInputFieldInitialFormValue } from 'features/nodes/hooks/useInputFieldInitialFormValue';
import type { NodeFieldElement } from 'features/nodes/types/workflow';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

type Props = {
  element: NodeFieldElement;
};

export const NodeFieldElementResetToInitialValueIconButton = memo(({ element }: Props) => {
  const { t } = useTranslation();
  const { id, data } = element;
  const { nodeId, fieldName } = data.fieldIdentifier;
  const { isValueChanged, resetToInitialValue } = useInputFieldInitialFormValue(id, nodeId, fieldName);

  if (!isValueChanged) {
    return null;
  }

  return (
    <IconButton
      variant="link"
      size="sm"
      alignSelf="stretch"
      tooltip={t('nodes.resetToDefaultValue')}
      aria-label={t('nodes.resetToDefaultValue')}
      icon={<PiArrowCounterClockwiseBold />}
      onClick={resetToInitialValue}
      isDisabled={!isValueChanged}
    />
  );
});

NodeFieldElementResetToInitialValueIconButton.displayName = 'NodeFieldElementResetToInitialValueIconButton';
