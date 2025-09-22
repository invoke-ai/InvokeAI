import { Box, Button, Flex, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import {
  canvasWorkflowCleared,
  selectCanvasWorkflow,
  selectCanvasWorkflowSlice,
} from 'features/controlLayers/store/canvasWorkflowSlice';
import { selectIsApiBaseModel, selectIsCogView4, selectIsSDXL } from 'features/controlLayers/store/paramsSlice';
import { useWorkflowLibraryModal } from 'features/nodes/store/workflowLibraryModal';
import { Prompts } from 'features/parameters/components/Prompts/Prompts';
import { AdvancedSettingsAccordion } from 'features/settingsAccordions/components/AdvancedSettingsAccordion/AdvancedSettingsAccordion';
import { CompositingSettingsAccordion } from 'features/settingsAccordions/components/CompositingSettingsAccordion/CompositingSettingsAccordion';
import { GenerationSettingsAccordion } from 'features/settingsAccordions/components/GenerationSettingsAccordion/GenerationSettingsAccordion';
import { CanvasTabImageSettingsAccordion } from 'features/settingsAccordions/components/ImageSettingsAccordion/CanvasTabImageSettingsAccordion';
import { RefinerSettingsAccordion } from 'features/settingsAccordions/components/RefinerSettingsAccordion/RefinerSettingsAccordion';
import { StylePresetMenu } from 'features/stylePresets/components/StylePresetMenu';
import { StylePresetMenuTrigger } from 'features/stylePresets/components/StylePresetMenuTrigger';
import { $isStylePresetsMenuOpen } from 'features/stylePresets/store/stylePresetSlice';
import { toast } from 'features/toast/toast';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { CSSProperties } from 'react';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

const overlayScrollbarsStyles: CSSProperties = {
  height: '100%',
  width: '100%',
};

export const ParametersPanelCanvas = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const isSDXL = useAppSelector(selectIsSDXL);
  const isCogview4 = useAppSelector(selectIsCogView4);
  const isStylePresetsMenuOpen = useStore($isStylePresetsMenuOpen);

  const isApiModel = useAppSelector(selectIsApiBaseModel);
  const workflowState = useAppSelector(selectCanvasWorkflowSlice);
  const workflowLibraryModal = useWorkflowLibraryModal();
  const isProcessingRef = useRef(false);

  const handleChangeWorkflow = useCallback(() => {
    workflowLibraryModal.open({
      mode: 'canvas',
      onSelect: async (workflowId: string) => {
        if (isProcessingRef.current) {
          return;
        }
        isProcessingRef.current = true;
        const result = await dispatch(selectCanvasWorkflow(workflowId));
        if (selectCanvasWorkflow.fulfilled.match(result)) {
          toast({ status: 'success', title: t('controlLayers.canvasWorkflowSelected') });
          workflowLibraryModal.close();
        } else {
          const message = result.payload ?? result.error.message ?? t('common.error');
          toast({ status: 'error', title: message });
        }
        isProcessingRef.current = false;
      },
    });
  }, [dispatch, t, workflowLibraryModal]);

  const handleClearWorkflow = useCallback(() => {
    dispatch(canvasWorkflowCleared());
  }, [dispatch]);

  if (workflowState.workflow) {
    return (
      <Flex w="full" h="full" flexDir="column" alignItems="center" justifyContent="center" gap={4} p={4}>
        <Text fontSize="lg" fontWeight="semibold" textAlign="center">
          {workflowState.workflow.name || t('controlLayers.canvasWorkflowLabel')}
        </Text>
        {workflowState.workflow.description && (
          <Text fontSize="sm" textAlign="center" color="base.300">
            {workflowState.workflow.description}
          </Text>
        )}
        {workflowState.error && (
          <Text fontSize="sm" color="invokeRed.300" textAlign="center">
            {workflowState.error}
          </Text>
        )}
        <Flex gap={2}>
          <Button size="sm" onClick={handleChangeWorkflow} isDisabled={workflowState.status === 'loading'}>
            {t('controlLayers.canvasWorkflowChangeButton')}
          </Button>
          <Button size="sm" variant="outline" onClick={handleClearWorkflow}>
            {t('common.clear')}
          </Button>
        </Flex>
      </Flex>
    );
  }

  return (
    <Flex w="full" h="full" flexDir="column" gap={2}>
      <StylePresetMenuTrigger />
      <Flex w="full" h="full" position="relative">
        <Box position="absolute" top={0} left={0} right={0} bottom={0}>
          {isStylePresetsMenuOpen && (
            <OverlayScrollbarsComponent defer style={overlayScrollbarsStyles} options={overlayScrollbarsParams.options}>
              <Flex gap={2} flexDirection="column" h="full" w="full">
                <StylePresetMenu />
              </Flex>
            </OverlayScrollbarsComponent>
          )}
          <OverlayScrollbarsComponent defer style={overlayScrollbarsStyles} options={overlayScrollbarsParams.options}>
            <Flex gap={2} flexDirection="column" h="full" w="full">
              <Prompts />
              <CanvasTabImageSettingsAccordion />
              <GenerationSettingsAccordion />
              {!isApiModel && <CompositingSettingsAccordion />}
              {isSDXL && <RefinerSettingsAccordion />}
              {!isCogview4 && !isApiModel && <AdvancedSettingsAccordion />}
            </Flex>
          </OverlayScrollbarsComponent>
        </Box>
      </Flex>
    </Flex>
  );
});

ParametersPanelCanvas.displayName = 'ParametersPanelCanvas';
