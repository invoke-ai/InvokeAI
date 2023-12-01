import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  FormControl,
  FormLabel,
  Input,
  useDisclosure,
} from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import { useSaveWorkflowAs } from 'features/workflowLibrary/hooks/useDuplicateWorkflow';
import { ChangeEvent, memo, useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FaClone } from 'react-icons/fa';

const SaveWorkflowAsButton = () => {
  const currentName = useAppSelector((state) => state.nodes.workflow.name);
  const { t } = useTranslation();
  const { saveWorkflowAs, isLoading } = useSaveWorkflowAs();
  const [name, setName] = useState(currentName.trim());
  const { isOpen, onOpen, onClose } = useDisclosure();
  const inputRef = useRef<HTMLInputElement>(null);

  const onSave = useCallback(async () => {
    saveWorkflowAs({ name, onSuccess: onClose, onError: onClose });
  }, [name, onClose, saveWorkflowAs]);

  const onChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setName(e.target.value);
  }, []);

  return (
    <>
      <IAIIconButton
        icon={<FaClone />}
        onClick={onOpen}
        isLoading={isLoading}
        tooltip={t('workflows.saveWorkflowAs')}
        aria-label={t('workflows.saveWorkflowAs')}
      />
      <AlertDialog
        isOpen={isOpen}
        onClose={onClose}
        leastDestructiveRef={inputRef}
        isCentered
      >
        <AlertDialogOverlay>
          <AlertDialogContent>
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              {t('workflows.saveWorkflowAs')}
            </AlertDialogHeader>

            <AlertDialogBody>
              <FormControl>
                <FormLabel>{t('workflows.workflowName')}</FormLabel>
                <Input
                  ref={inputRef}
                  value={name}
                  onChange={onChange}
                  placeholder={t('workflows.workflowName')}
                />
              </FormControl>
            </AlertDialogBody>

            <AlertDialogFooter>
              <IAIButton onClick={onClose}>{t('common.cancel')}</IAIButton>
              <IAIButton colorScheme="accent" onClick={onSave} ml={3}>
                {t('common.saveAs')}
              </IAIButton>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    </>
  );
};

export default memo(SaveWorkflowAsButton);
