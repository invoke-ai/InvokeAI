import { MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { $hasTemplates } from 'features/nodes/store/nodesSlice';
import { useLoadWorkflowWithDialog } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFlowArrowBold } from 'react-icons/pi';

export const ContextMenuItemLoadWorkflow = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();
  const loadWorkflowWithDialog = useLoadWorkflowWithDialog();
  const hasTemplates = useStore($hasTemplates);

  const onClick = useCallback(() => {
      loadWorkflowWithDialog({ type: 'image', data: imageDTO.image_name });
  }, [loadWorkflowWithDialog, imageDTO]);

  const isDisabled = useMemo(() => {
      return !imageDTO.has_workflow || !hasTemplates;
  }, [imageDTO, hasTemplates]);

  return (
    <MenuItem icon={<PiFlowArrowBold />} onClickCapture={onClick} isDisabled={isDisabled}>
      {t('nodes.loadWorkflow')}
    </MenuItem>
  );
});

ContextMenuItemLoadWorkflow.displayName = 'ContextMenuItemLoadWorkflow';
