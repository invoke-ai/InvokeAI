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
import { t } from 'i18next';
import type { ChangeEvent } from 'react';
import { useCallback, useRef, useState } from 'react';

import { $workflowCategories } from '../../../../app/store/nanostores/workflowCategories';
import { useAppSelector } from '../../../../app/store/storeHooks';
import { useSaveWorkflowAs } from '../../hooks/useSaveWorkflowAs';
import { getWorkflowCopyName } from '../../util/getWorkflowCopyName';

export const SaveWorkflowAsDialog = ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => {
  const currentName = useAppSelector((s) => s.workflow.name);
  const [name, setName] = useState(currentName.length ? getWorkflowCopyName(currentName) : '');
  const [saveToProject, setSaveToProject] = useState(false);

  const workflowCategories = useStore($workflowCategories);

  const { saveWorkflowAs } = useSaveWorkflowAs();

  const cancelRef = useRef<HTMLButtonElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const onChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setName(e.target.value);
  }, []);

  const onChangeCheckbox = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setSaveToProject(e.target.checked);
  }, []);

  const clearAndClose = useCallback(() => {
    onClose();
    setName('');
    setSaveToProject(false);
  }, [onClose]);

  const onSave = useCallback(async () => {
    const category = saveToProject ? 'project' : 'user';
    await saveWorkflowAs({
      name,
      category,
      onSuccess: clearAndClose,
      onError: clearAndClose,
    });
  }, [name, saveWorkflowAs, saveToProject, clearAndClose]);

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
                <Input ref={inputRef} value={name} onChange={onChange} placeholder={t('workflows.workflowName')} />
                {workflowCategories.includes('project') && (
                  <Checkbox isChecked={saveToProject} onChange={onChangeCheckbox}>
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
            <Button colorScheme="invokeBlue" onClick={onSave} ml={3} isDisabled={!name || !name.length}>
              {t('common.saveAs')}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialogOverlay>
    </AlertDialog>
  );
};
