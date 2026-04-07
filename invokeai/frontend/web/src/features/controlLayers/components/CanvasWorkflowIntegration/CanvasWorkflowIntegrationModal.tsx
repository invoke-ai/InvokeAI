import {
  Button,
  ButtonGroup,
  Flex,
  Heading,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Spacer,
  Spinner,
  Text,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  canvasWorkflowIntegrationClosed,
  selectCanvasWorkflowIntegrationIsOpen,
  selectCanvasWorkflowIntegrationIsProcessing,
  selectCanvasWorkflowIntegrationSelectedWorkflowId,
} from 'features/controlLayers/store/canvasWorkflowIntegrationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { CanvasWorkflowIntegrationParameterPanel } from './CanvasWorkflowIntegrationParameterPanel';
import { CanvasWorkflowIntegrationWorkflowSelector } from './CanvasWorkflowIntegrationWorkflowSelector';
import { useCanvasWorkflowIntegrationExecute } from './useCanvasWorkflowIntegrationExecute';

export const CanvasWorkflowIntegrationModal = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const isOpen = useAppSelector(selectCanvasWorkflowIntegrationIsOpen);
  const isProcessing = useAppSelector(selectCanvasWorkflowIntegrationIsProcessing);
  const selectedWorkflowId = useAppSelector(selectCanvasWorkflowIntegrationSelectedWorkflowId);

  const { execute, canExecute } = useCanvasWorkflowIntegrationExecute();

  const onClose = useCallback(() => {
    if (!isProcessing) {
      dispatch(canvasWorkflowIntegrationClosed());
    }
  }, [dispatch, isProcessing]);

  const onExecute = useCallback(() => {
    execute();
  }, [execute]);

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="xl" isCentered>
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>
          <Heading size="md">{t('controlLayers.workflowIntegration.title')}</Heading>
        </ModalHeader>
        <ModalCloseButton isDisabled={isProcessing} />

        <ModalBody>
          <Flex direction="column" gap={4}>
            <Text fontSize="sm" color="base.400">
              {t('controlLayers.workflowIntegration.description')}
            </Text>

            <CanvasWorkflowIntegrationWorkflowSelector />

            {selectedWorkflowId && <CanvasWorkflowIntegrationParameterPanel />}
          </Flex>
        </ModalBody>

        <ModalFooter>
          <ButtonGroup>
            <Button variant="ghost" onClick={onClose} isDisabled={isProcessing}>
              {t('common.cancel')}
            </Button>
            <Spacer />
            <Button
              onClick={onExecute}
              isDisabled={!canExecute || isProcessing}
              loadingText={t('controlLayers.workflowIntegration.executing')}
            >
              {isProcessing && <Spinner size="sm" mr={2} />}
              {t('controlLayers.workflowIntegration.execute')}
            </Button>
          </ButtonGroup>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
});

CanvasWorkflowIntegrationModal.displayName = 'CanvasWorkflowIntegrationModal';
