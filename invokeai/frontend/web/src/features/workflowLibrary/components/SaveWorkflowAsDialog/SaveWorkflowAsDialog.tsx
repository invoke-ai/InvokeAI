import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  Button,
  Checkbox,
  Flex,
  FormControl,
  FormLabel,
  Input,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $workflowCategories } from 'app/store/nanostores/workflowCategories';
import { useSaveWorkflowAsDialog } from 'features/workflowLibrary/components/SaveWorkflowAsDialog/useSaveWorkflowAsDialog';
import { useSaveWorkflowAs } from 'features/workflowLibrary/hooks/useSaveWorkflowAs';
import { t } from 'i18next';
import type { ChangeEvent } from 'react';
import { useCallback, useRef } from 'react';

export const SaveWorkflowAsDialog = () => {
  const { isOpen, onClose, workflowName, setWorkflowName, shouldSaveToProject, setShouldSaveToProject } =
    useSaveWorkflowAsDialog();

  const workflowCategories = useStore($workflowCategories);

  const { saveWorkflowAs } = useSaveWorkflowAs();

  const cancelRef = useRef<HTMLButtonElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      setWorkflowName(e.target.value);
    },
    [setWorkflowName]
  );

  const onChangeCheckbox = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      setShouldSaveToProject(e.target.checked);
    },
    [setShouldSaveToProject]
  );

  const clearAndClose = useCallback(() => {
    onClose();
  }, [onClose]);

  const onSave = useCallback(async () => {
    const category = shouldSaveToProject ? 'project' : 'user';
    await saveWorkflowAs({
      name: workflowName,
      category,
      onSuccess: clearAndClose,
      onError: clearAndClose,
    });
  }, [workflowName, saveWorkflowAs, shouldSaveToProject, clearAndClose]);

  return (
    <AlertDialog isOpen={isOpen} onClose={onClose} leastDestructiveRef={cancelRef} isCentered={true}>
      <AlertDialogOverlay>
        <AlertDialogContent>
          <AlertDialogHeader fontSize="lg" fontWeight="bold">
            {t('workflows.saveWorkflowAs')}
          </AlertDialogHeader>

          <AlertDialogBody>
            <FormControl alignItems="flex-start">
              <FormLabel mt="2">{t('workflows.workflowName')}</FormLabel>
              <Flex flexDir="column" width="full" gap="2">
                <Input
                  ref={inputRef}
                  value={workflowName}
                  onChange={onChange}
                  placeholder={t('workflows.workflowName')}
                />
                {workflowCategories.includes('project') && (
                  <Checkbox isChecked={shouldSaveToProject} onChange={onChangeCheckbox}>
                    <FormLabel>{t('workflows.saveWorkflowToProject')}</FormLabel>
                  </Checkbox>
                )}
              </Flex>
            </FormControl>
          </AlertDialogBody>

          <AlertDialogFooter>
            <Button ref={cancelRef} onClick={clearAndClose}>
              {t('common.cancel')}
            </Button>
            <Button colorScheme="invokeBlue" onClick={onSave} ml={3} isDisabled={!workflowName || !workflowName.length}>
              {t('common.saveAs')}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialogOverlay>
    </AlertDialog>
  );
};
