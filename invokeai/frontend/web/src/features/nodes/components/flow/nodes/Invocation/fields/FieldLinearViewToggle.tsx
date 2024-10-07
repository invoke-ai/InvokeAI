import { IconButton } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useFieldValue } from 'features/nodes/hooks/useFieldValue';
import {
  selectWorkflowSlice,
  workflowExposedFieldAdded,
  workflowExposedFieldRemoved,
} from 'features/nodes/store/workflowSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMinusBold, PiPlusBold } from 'react-icons/pi';

type Props = {
  nodeId: string;
  fieldName: string;
};

const FieldLinearViewToggle = ({ nodeId, fieldName }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const value = useFieldValue(nodeId, fieldName);
  const selectIsExposed = useMemo(
    () =>
      createSelector(selectWorkflowSlice, (workflow) => {
        return Boolean(workflow.exposedFields.find((f) => f.nodeId === nodeId && f.fieldName === fieldName));
      }),
    [fieldName, nodeId]
  );

  const isExposed = useAppSelector(selectIsExposed);

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
};

export default memo(FieldLinearViewToggle);
