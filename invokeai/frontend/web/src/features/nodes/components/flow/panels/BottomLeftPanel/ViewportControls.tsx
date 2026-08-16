import { ButtonGroup, IconButton } from '@invoke-ai/ui-library';
import { useReactFlow } from '@xyflow/react';
import { logger } from 'app/logging/logger';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowName } from 'features/nodes/store/selectors';
import {
  selectShouldShowMinimapPanel,
  shouldShowMinimapPanelChanged,
} from 'features/nodes/store/workflowSettingsSlice';
import { exportWorkflowAsPng } from 'features/nodes/util/workflowImageExport';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiCameraBold,
  PiFrameCornersBold,
  PiMagnifyingGlassMinusBold,
  PiMagnifyingGlassPlusBold,
  PiMapPinBold,
} from 'react-icons/pi';

import { AutoLayoutPopover } from './AutoLayoutPopover';

const log = logger('workflows');

const ViewportControls = () => {
  const { t } = useTranslation();
  const { zoomIn, zoomOut, fitView, getNodes, getNodesBounds } = useReactFlow();
  const dispatch = useAppDispatch();
  const shouldShowMinimapPanel = useAppSelector(selectShouldShowMinimapPanel);
  const workflowName = useAppSelector(selectWorkflowName);
  const fallbackWorkflowName = t('workflows.unnamedWorkflow');
  const [isExportingWorkflow, setIsExportingWorkflow] = useState(false);

  const handleClickedZoomIn = useCallback(() => {
    zoomIn({ duration: 300 });
  }, [zoomIn]);

  const handleClickedZoomOut = useCallback(() => {
    zoomOut({ duration: 300 });
  }, [zoomOut]);

  const handleClickedFitView = useCallback(() => {
    fitView({ duration: 300 });
  }, [fitView]);

  const handleClickedToggleMiniMapPanel = useCallback(() => {
    dispatch(shouldShowMinimapPanelChanged(!shouldShowMinimapPanel));
  }, [shouldShowMinimapPanel, dispatch]);

  const handleWorkflowImageExportError = useCallback(
    (error?: unknown) => {
      if (error) {
        log.error({ error: error instanceof Error ? error.message : String(error) }, 'Workflow image export failed');
      }
      toast({
        id: 'DOWNLOAD_WORKFLOW_IMAGE_ERROR',
        status: 'error',
        description: t('nodes.downloadWorkflowImageError'),
      });
    },
    [t]
  );

  const handleClickedExportWorkflow = useCallback(() => {
    if (isExportingWorkflow) {
      return;
    }

    const flowElement = document.querySelector<HTMLElement>('#workflow-editor');
    if (!flowElement) {
      handleWorkflowImageExportError();
      return;
    }

    setIsExportingWorkflow(true);
    void new Promise<void>((resolve) => {
      requestAnimationFrame(() => resolve());
    })
      .then(() =>
        exportWorkflowAsPng({
          flowElement,
          bounds: getNodesBounds(getNodes()),
          workflowName,
          fallbackWorkflowName,
        })
      )
      .catch(handleWorkflowImageExportError)
      .finally(() => setIsExportingWorkflow(false));
  }, [
    fallbackWorkflowName,
    getNodes,
    getNodesBounds,
    handleWorkflowImageExportError,
    isExportingWorkflow,
    workflowName,
  ]);

  return (
    <ButtonGroup orientation="vertical">
      <IconButton
        tooltip={t('nodes.zoomInNodes')}
        aria-label={t('nodes.zoomInNodes')}
        onClick={handleClickedZoomIn}
        icon={<PiMagnifyingGlassPlusBold />}
      />
      <IconButton
        tooltip={t('nodes.zoomOutNodes')}
        aria-label={t('nodes.zoomOutNodes')}
        onClick={handleClickedZoomOut}
        icon={<PiMagnifyingGlassMinusBold />}
      />
      <IconButton
        tooltip={t('nodes.fitViewportNodes')}
        aria-label={t('nodes.fitViewportNodes')}
        onClick={handleClickedFitView}
        icon={<PiFrameCornersBold />}
      />
      <AutoLayoutPopover />
      <IconButton
        tooltip={t('nodes.downloadWorkflowImage')}
        aria-label={t('nodes.downloadWorkflowImage')}
        isDisabled={isExportingWorkflow}
        isLoading={isExportingWorkflow}
        onClick={handleClickedExportWorkflow}
        icon={<PiCameraBold />}
      />
      <IconButton
        tooltip={shouldShowMinimapPanel ? t('nodes.hideMinimapnodes') : t('nodes.showMinimapnodes')}
        aria-label={shouldShowMinimapPanel ? t('nodes.hideMinimapnodes') : t('nodes.showMinimapnodes')}
        isChecked={shouldShowMinimapPanel}
        onClick={handleClickedToggleMiniMapPanel}
        icon={<PiMapPinBold />}
      />
    </ButtonGroup>
  );
};

export default memo(ViewportControls);
