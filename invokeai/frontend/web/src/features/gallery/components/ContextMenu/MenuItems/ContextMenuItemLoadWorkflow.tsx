import { MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useItemDTOContext } from 'features/gallery/contexts/ItemDTOContext';
import { $hasTemplates } from 'features/nodes/store/nodesSlice';
import { useLoadWorkflowWithDialog } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFlowArrowBold } from 'react-icons/pi';
import { isImageDTO } from 'services/api/types';

export const ContextMenuItemLoadWorkflow = memo(() => {
  const { t } = useTranslation();
  const itemDTO = useItemDTOContext();
  const loadWorkflowWithDialog = useLoadWorkflowWithDialog();
  const hasTemplates = useStore($hasTemplates);

  const onClick = useCallback(() => {
    if (isImageDTO(itemDTO)) {
      loadWorkflowWithDialog({ type: 'image', data: itemDTO.image_name });
    } else {
      // loadWorkflowWithDialog({ type: 'video', data: itemDTO.video_id });
    }
  }, [loadWorkflowWithDialog, itemDTO]);

  const isDisabled = useMemo(() => {
    if (isImageDTO(itemDTO)) {
      return !itemDTO.has_workflow || !hasTemplates;
    }
    return false
  }, [itemDTO, hasTemplates]);

  return (
    <MenuItem icon={<PiFlowArrowBold />} onClickCapture={onClick} isDisabled={isDisabled}>
      {t('nodes.loadWorkflow')}
    </MenuItem>
  );
});

ContextMenuItemLoadWorkflow.displayName = 'ContextMenuItemLoadWorkflow';
