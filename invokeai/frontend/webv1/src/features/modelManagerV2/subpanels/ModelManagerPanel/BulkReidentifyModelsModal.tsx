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

type BulkReidentifyModelsModalProps = {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  modelCount: number;
  isReidentifying?: boolean;
};

export const BulkReidentifyModelsModal = memo(
  ({ isOpen, onClose, onConfirm, modelCount, isReidentifying = false }: BulkReidentifyModelsModalProps) => {
    const { t } = useTranslation();
    const cancelRef = useRef<HTMLButtonElement>(null);

    return (
      <AlertDialog isOpen={isOpen} onClose={onClose} leastDestructiveRef={cancelRef} isCentered>
        <AlertDialogOverlay>
          <AlertDialogContent>
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              {t('modelManager.reidentifyModels', {
                count: modelCount,
                defaultValue: 'Reidentify Models',
              })}
            </AlertDialogHeader>

            <AlertDialogBody>
              <Flex flexDir="column" gap={3}>
                <Text>
                  {t('modelManager.reidentifyModelsConfirm', {
                    count: modelCount,
                    defaultValue: `Are you sure you want to reidentify ${modelCount} model(s)? This will re-probe their weights files to determine the correct format and settings.`,
                  })}
                </Text>
                <Text fontWeight="semibold" color="warning.400">
                  {t('modelManager.reidentifyWarning', {
                    defaultValue: 'This will reset any custom settings you may have applied to these models.',
                  })}
                </Text>
              </Flex>
            </AlertDialogBody>

            <AlertDialogFooter>
              <Button ref={cancelRef} onClick={onClose} isDisabled={isReidentifying}>
                {t('common.cancel')}
              </Button>
              <Button colorScheme="warning" onClick={onConfirm} ml={3} isLoading={isReidentifying}>
                {t('modelManager.reidentify')}
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    );
  }
);

BulkReidentifyModelsModal.displayName = 'BulkReidentifyModelsModal';
