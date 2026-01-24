import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  Button,
  Flex,
  Text,
} from '@invoke-ai/ui-library';
import { memo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

type BulkDeleteModelsModalProps = {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  modelCount: number;
  isDeleting?: boolean;
};

export const BulkDeleteModelsModal = memo(
  ({ isOpen, onClose, onConfirm, modelCount, isDeleting = false }: BulkDeleteModelsModalProps) => {
    const { t } = useTranslation();
    const cancelRef = useRef<HTMLButtonElement>(null);

    return (
      <AlertDialog isOpen={isOpen} onClose={onClose} leastDestructiveRef={cancelRef} isCentered>
        <AlertDialogOverlay>
          <AlertDialogContent>
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              {t('modelManager.deleteModels', { count: modelCount })}
            </AlertDialogHeader>

            <AlertDialogBody>
              <Flex flexDir="column" gap={3}>
                <Text>
                  {t('modelManager.deleteModelsConfirm', {
                    count: modelCount,
                    defaultValue: `Are you sure you want to delete ${modelCount} model(s)? This action cannot be undone.`,
                  })}
                </Text>
                <Text fontWeight="semibold" color="error.400">
                  {t('modelManager.deleteWarning', {
                    defaultValue: 'Models in your Invoke models directory will be permanently deleted from disk.',
                  })}
                </Text>
              </Flex>
            </AlertDialogBody>

            <AlertDialogFooter>
              <Button ref={cancelRef} onClick={onClose} isDisabled={isDeleting}>
                {t('common.cancel')}
              </Button>
              <Button colorScheme="error" onClick={onConfirm} ml={3} isLoading={isDeleting}>
                {t('common.delete')}
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    );
  }
);

BulkDeleteModelsModal.displayName = 'BulkDeleteModelsModal';
