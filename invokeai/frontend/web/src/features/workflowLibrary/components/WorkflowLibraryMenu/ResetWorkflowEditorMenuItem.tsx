import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  Button,
  Flex,
  MenuItem,
  Text,
  useDisclosure,
} from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { FaTrash } from 'react-icons/fa';

const ResetWorkflowEditorMenuItem = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const cancelRef = useRef<HTMLButtonElement | null>(null);

  const handleConfirmClear = useCallback(() => {
    dispatch(nodeEditorReset());

    dispatch(
      addToast(
        makeToast({
          title: t('workflows.workflowEditorReset'),
          status: 'success',
        })
      )
    );

    onClose();
  }, [dispatch, t, onClose]);

  return (
    <>
      <MenuItem
        as="button"
        icon={<FaTrash />}
        sx={{ color: 'error.600', _dark: { color: 'error.300' } }}
        onClick={onOpen}
      >
        {t('nodes.resetWorkflow')}
      </MenuItem>

      <AlertDialog
        isOpen={isOpen}
        onClose={onClose}
        leastDestructiveRef={cancelRef}
        isCentered
      >
        <AlertDialogOverlay />

        <AlertDialogContent>
          <AlertDialogHeader fontSize="lg" fontWeight="bold">
            {t('nodes.resetWorkflow')}
          </AlertDialogHeader>

          <AlertDialogBody py={4}>
            <Flex flexDir="column" gap={2}>
              <Text>{t('nodes.resetWorkflowDesc')}</Text>
              <Text variant="subtext">{t('nodes.resetWorkflowDesc2')}</Text>
            </Flex>
          </AlertDialogBody>

          <AlertDialogFooter>
            <Button ref={cancelRef} onClick={onClose}>
              {t('common.cancel')}
            </Button>
            <Button colorScheme="error" ml={3} onClick={handleConfirmClear}>
              {t('common.accept')}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
};

export default memo(ResetWorkflowEditorMenuItem);
