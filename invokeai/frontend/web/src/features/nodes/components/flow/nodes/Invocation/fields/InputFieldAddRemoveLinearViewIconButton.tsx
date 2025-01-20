import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useInputFieldIsExposed } from 'features/nodes/hooks/useInputFieldIsExposed';
import { useInputFieldValue } from 'features/nodes/hooks/useInputFieldValue';
import { workflowExposedFieldAdded, workflowExposedFieldRemoved } from 'features/nodes/store/workflowSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMinusBold, PiPlusBold } from 'react-icons/pi';

type Props = {
  nodeId: string;
  fieldName: string;
};

export const InputFieldAddRemoveLinearViewIconButton = memo(({ nodeId, fieldName }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const value = useInputFieldValue(nodeId, fieldName);
  const isExposed = useInputFieldIsExposed(nodeId, fieldName);

  const handleExposeField = useCallback(() => {
    dispatch(workflowExposedFieldAdded({ nodeId, fieldName, value }));
  }, [dispatch, fieldName, nodeId, value]);

  const handleUnexposeField = useCallback(() => {
    dispatch(workflowExposedFieldRemoved({ nodeId, fieldName }));
  }, [dispatch, fieldName, nodeId]);

  if (!isExposed) {
    return (
      <IconButton
        variant="ghost"
        tooltip={t('nodes.addLinearView')}
        aria-label={t('nodes.addLinearView')}
        icon={<PiPlusBold />}
        onClick={handleExposeField}
        pointerEvents="auto"
        size="xs"
      />
    );
  } else {
    return (
      <IconButton
        variant="ghost"
        tooltip={t('nodes.removeLinearView')}
        aria-label={t('nodes.removeLinearView')}
        icon={<PiMinusBold />}
        onClick={handleUnexposeField}
        pointerEvents="auto"
        size="xs"
      />
    );
  }
});

InputFieldAddRemoveLinearViewIconButton.displayName = 'InputFieldAddRemoveLinearViewIconButton';
