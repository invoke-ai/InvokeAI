import { MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { $hasTemplates } from 'features/nodes/store/nodesSlice';
import { useLoadWorkflowWithDialog } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFlowArrowBold } from 'react-icons/pi';

export const ImageMenuItemLoadWorkflow = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();
  const loadWorkflowWithDialog = useLoadWorkflowWithDialog();
  const hasTemplates = useStore($hasTemplates);

  const onClick = useCallback(() => {
    loadWorkflowWithDialog({ type: 'image', data: imageDTO.image_name });
  }, [loadWorkflowWithDialog, imageDTO.image_name]);

  return (
    <MenuItem icon={<PiFlowArrowBold />} onClickCapture={onClick} isDisabled={!imageDTO.has_workflow || !hasTemplates}>
      {t('nodes.loadWorkflow')}
    </MenuItem>
  );
});

ImageMenuItemLoadWorkflow.displayName = 'ImageMenuItemLoadWorkflow';
