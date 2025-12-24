import {
  Button,
  Flex,
  FormControl,
  FormErrorMessage,
  FormLabel,
  Input,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Text,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { isExternalModel } from 'features/modelManagerV2/subpanels/ModelPanel/isExternalModel';
import { toast } from 'features/toast/toast';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFolderOpenFill } from 'react-icons/pi';
import { useUpdateModelMutation } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

interface Props {
  modelConfig: AnyModelConfig;
}

export const ModelUpdatePathButton = memo(({ modelConfig }: Props) => {
  const { t } = useTranslation();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [updateModel, { isLoading }] = useUpdateModelMutation();
  const [newPath, setNewPath] = useState(modelConfig.path);

  const handleOpen = useCallback(() => {
    setNewPath(modelConfig.path);
    onOpen();
  }, [modelConfig.path, onOpen]);

  const handlePathChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setNewPath(e.target.value);
  }, []);

  // Validate that the new path is still a valid external model path format
  const isValidPath = useMemo(() => isExternalModel(newPath.trim()), [newPath]);

  const handleSubmit = useCallback(() => {
    if (!newPath.trim() || newPath === modelConfig.path || !isValidPath) {
      onClose();
      return;
    }

    updateModel({
      key: modelConfig.key,
      body: { path: newPath.trim() },
    })
      .unwrap()
      .then(() => {
        toast({
          id: 'MODEL_PATH_UPDATED',
          title: t('modelManager.pathUpdated'),
          status: 'success',
        });
        onClose();
      })
      .catch(() => {
        toast({
          id: 'MODEL_PATH_UPDATE_FAILED',
          title: t('modelManager.pathUpdateFailed'),
          status: 'error',
        });
      });
  }, [newPath, modelConfig.path, modelConfig.key, updateModel, onClose, t, isValidPath]);

  const hasChanges = newPath.trim() !== modelConfig.path;
  const showError = hasChanges && !isValidPath;

  return (
    <>
      <Button
        onClick={handleOpen}
        size="sm"
        aria-label={t('modelManager.updatePathTooltip')}
        tooltip={t('modelManager.updatePathTooltip')}
        flexShrink={0}
        leftIcon={<PiFolderOpenFill />}
      >
        {t('modelManager.updatePath')}
      </Button>
      <Modal isOpen={isOpen} onClose={onClose} isCentered size="lg" useInert={false}>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>{t('modelManager.updatePath')}</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <Flex flexDirection="column" gap={4}>
              <Text fontSize="sm" color="base.400">
                {t('modelManager.updatePathDescription')}
              </Text>
              <FormControl>
                <FormLabel>{t('modelManager.currentPath')}</FormLabel>
                <Text fontSize="sm" color="base.300" wordBreak="break-all">
                  {modelConfig.path}
                </Text>
              </FormControl>
              <FormControl isInvalid={showError}>
                <FormLabel>{t('modelManager.newPath')}</FormLabel>
                <Input value={newPath} onChange={handlePathChange} placeholder={t('modelManager.newPathPlaceholder')} />
                {showError && <FormErrorMessage>{t('modelManager.invalidPathFormat')}</FormErrorMessage>}
              </FormControl>
            </Flex>
          </ModalBody>
          <ModalFooter>
            <Flex gap={2}>
              <Button variant="ghost" onClick={onClose}>
                {t('common.cancel')}
              </Button>
              <Button
                colorScheme="invokeYellow"
                onClick={handleSubmit}
                isLoading={isLoading}
                isDisabled={!hasChanges || showError}
              >
                {t('common.save')}
              </Button>
            </Flex>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </>
  );
});

ModelUpdatePathButton.displayName = 'ModelUpdatePathButton';
