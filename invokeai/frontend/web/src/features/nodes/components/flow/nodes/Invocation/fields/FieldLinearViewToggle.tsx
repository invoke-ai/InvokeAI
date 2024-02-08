import { IconButton } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
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
  isHovered: boolean;
};

const FieldLinearViewToggle = ({ nodeId, fieldName, isHovered }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const selectIsExposed = useMemo(
    () =>
      createSelector(selectWorkflowSlice, (workflow) => {
        return Boolean(workflow.exposedFields.find((f) => f.nodeId === nodeId && f.fieldName === fieldName));
      }),
    [fieldName, nodeId]
  );

  const isExposed = useAppSelector(selectIsExposed);

  const handleExposeField = useCallback(() => {
    dispatch(workflowExposedFieldAdded({ nodeId, fieldName }));
  }, [dispatch, fieldName, nodeId]);

  const handleUnexposeField = useCallback(() => {
    dispatch(workflowExposedFieldRemoved({ nodeId, fieldName }));
  }, [dispatch, fieldName, nodeId]);

  const ToggleButton = useMemo(() => {
    if (!isHovered) {
      return null;
    } else if (!isExposed) {
      return (
        <IconButton
          mx="2"
          tooltip={t('nodes.addLinearView')}
          aria-label={t('nodes.addLinearView')}
          icon={<PiPlusBold />}
          onClick={handleExposeField}
          pointerEvents="auto"
          size="xs"
        />
      );
    } else if (isExposed) {
      return (
        <IconButton
          mx="2"
          tooltip={t('nodes.removeLinearView')}
          aria-label={t('nodes.removeLinearView')}
          icon={<PiMinusBold />}
          onClick={handleUnexposeField}
          pointerEvents="auto"
          size="xs"
        />
      );
    }
  }, [isHovered, handleExposeField, handleUnexposeField, isExposed, t]);

  return ToggleButton;
};

export default memo(FieldLinearViewToggle);
