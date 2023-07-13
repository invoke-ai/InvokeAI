import { useDisclosure } from '@chakra-ui/react';
import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  Divider,
  Text,
  Button,
} from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { clearNodes } from 'features/nodes/store/nodesSlice';
import { makeToast } from 'app/components/Toaster';
import { addToast } from 'features/system/store/systemSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaTrash } from 'react-icons/fa';
import IAIIconButton from 'common/components/IAIIconButton';

const ClearNodesButton = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { isOpen, onOpen, onClose } = useDisclosure();

  const handleClearNodes = () => {
    onOpen();
  };

  const handleConfirmClear = () => {
    dispatch(clearNodes());

    dispatch(
      addToast(
        makeToast({
          title: t('toast.nodesCleared'),
          status: 'success',
        })
      )
    );

    onClose();
  };

  const handleCancelClear = () => {
    onClose();
  };

  return (
    <>
      <IAIIconButton
        icon={<FaTrash />}
        tooltip={t('nodes.clearNodes')}
        aria-label={t('nodes.clearNodes')}
        onClick={handleClearNodes}
      />

      <AlertDialog isOpen={isOpen} onClose={handleCancelClear} isCentered>
        <AlertDialogOverlay />

        <AlertDialogContent>
          <AlertDialogHeader fontSize="lg" fontWeight="bold">
            {t('nodes.clearNodes')}
          </AlertDialogHeader>

          <AlertDialogBody>
            <Text>{t('common.clearNodes')}</Text>
          </AlertDialogBody>

          <Divider />

          <AlertDialogFooter>
            <Button onClick={handleCancelClear}>{t('common.cancel')}</Button>
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
