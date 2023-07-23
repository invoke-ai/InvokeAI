import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  Button,
  Text,
  useDisclosure,
} from '@chakra-ui/react';
import { makeToast } from 'features/system/util/makeToast';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { addToast } from 'features/system/store/systemSlice';
import { memo, useRef, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaTrash } from 'react-icons/fa';

const ClearNodesButton = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const cancelRef = useRef<HTMLButtonElement | null>(null);

  const nodes = useAppSelector((state: RootState) => state.nodes.nodes);

  const handleConfirmClear = useCallback(() => {
    dispatch(nodeEditorReset());

    dispatch(
      addToast(
        makeToast({
          title: t('toast.nodesCleared'),
          status: 'success',
        })
      )
    );

    onClose();
  }, [dispatch, t, onClose]);

  return (
    <>
      <IAIIconButton
        icon={<FaTrash />}
        tooltip={t('nodes.clearNodes')}
        aria-label={t('nodes.clearNodes')}
        onClick={onOpen}
        isDisabled={nodes.length === 0}
      />

      <AlertDialog
        isOpen={isOpen}
        onClose={onClose}
        leastDestructiveRef={cancelRef}
        isCentered
      >
        <AlertDialogOverlay />

        <AlertDialogContent>
          <AlertDialogHeader fontSize="lg" fontWeight="bold">
            {t('nodes.clearNodes')}
          </AlertDialogHeader>

          <AlertDialogBody>
            <Text>{t('common.clearNodes')}</Text>
          </AlertDialogBody>

          <AlertDialogFooter>
            <Button ref={cancelRef} onClick={onClose}>
              {t('common.cancel')}
            </Button>
            <Button colorScheme="red" ml={3} onClick={handleConfirmClear}>
              {t('common.accept')}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
};

export default memo(ClearNodesButton);
