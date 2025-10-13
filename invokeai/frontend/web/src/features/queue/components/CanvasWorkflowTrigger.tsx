import { IconButton, Spinner, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectCanvasWorkflow, selectCanvasWorkflowSlice } from 'features/controlLayers/store/canvasWorkflowSlice';
import { useWorkflowLibraryModal } from 'features/nodes/store/workflowLibraryModal';
import { toast } from 'features/toast/toast';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFlowArrowBold } from 'react-icons/pi';

export const CanvasWorkflowTrigger = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const workflowLibraryModal = useWorkflowLibraryModal();
  const workflowState = useAppSelector(selectCanvasWorkflowSlice);
  const activeTab = useAppSelector(selectActiveTab);
  const isProcessingRef = useRef(false);

  const handleOpen = useCallback(() => {
    workflowLibraryModal.open({
      mode: 'canvas',
      onSelect: async (workflowId: string) => {
        if (isProcessingRef.current) {
          return;
        }
        isProcessingRef.current = true;
        const result = (await dispatch(selectCanvasWorkflow(workflowId))) as
          | ReturnType<typeof selectCanvasWorkflow.fulfilled>
          | ReturnType<typeof selectCanvasWorkflow.rejected>;
        if (selectCanvasWorkflow.fulfilled.match(result)) {
          toast({ status: 'success', title: t('controlLayers.canvasWorkflowSelected') });
          workflowLibraryModal.close();
        } else {
          const message = result.payload ?? result.error?.message ?? t('common.error');
          toast({ status: 'error', title: `${t('common.error')}: ${message}` });
        }
        isProcessingRef.current = false;
      },
    });
  }, [dispatch, t, workflowLibraryModal]);

  if (activeTab !== 'canvas') {
    return null;
  }

  return (
    <Tooltip
      label={
        workflowState.workflow
          ? t('controlLayers.changeCanvasWorkflowTooltip')
          : t('controlLayers.selectCanvasWorkflowTooltip')
      }
    >
      <IconButton
        aria-label={t('controlLayers.canvasWorkflowSelectButton')}
        size="lg"
        variant={workflowState.selectedWorkflowId ? 'solid' : 'outline'}
        colorScheme="invokeBlue"
        icon={workflowState.status === 'loading' ? <Spinner size="xs" /> : <PiFlowArrowBold />}
        onClick={handleOpen}
        isDisabled={workflowState.status === 'loading'}
      />
    </Tooltip>
  );
});

CanvasWorkflowTrigger.displayName = 'CanvasWorkflowTrigger';
