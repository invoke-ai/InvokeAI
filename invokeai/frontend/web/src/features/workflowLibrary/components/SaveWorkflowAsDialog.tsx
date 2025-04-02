import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  Button,
  Checkbox,
  Flex,
  FormControl,
  FormLabel,
  Input,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { deepClone } from 'common/util/deepClone';
import { $workflowLibraryCategoriesOptions } from 'features/nodes/store/workflowLibrarySlice';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { isDraftWorkflow, useCreateLibraryWorkflow } from 'features/workflowLibrary/hooks/useCreateNewWorkflow';
import { t } from 'i18next';
import { atom, computed } from 'nanostores';
import type { ChangeEvent, RefObject } from 'react';
import { memo, useCallback, useRef, useState } from 'react';
import { assert } from 'tsafe';

/**
 * The workflow to save as a new workflow.
 *
 * This state is used to determine whether or not the modal is open.
 */
const $workflowToSave = atom<WorkflowV3 | null>(null);

/**
 * Whether or not the modal is open. It is open if there is a workflow to save.
 *
 * The state is derived from the workflow to save.
 *
 * To open the modal, set the workflow to save to a workflow object.
 * To close the modal, set the workflow to save to null.
 */
const $isOpen = computed($workflowToSave, (val) => val !== null);

const getInitialName = (workflow: WorkflowV3): string => {
  if (!workflow.id) {
    // If the workflow has no ID, that means it's a new workflow that has never been saved to the server. In this case,
    // we should use whatever the user has entered in the workflow name field.
    return workflow.name;
  }
  // Otherwise, the workflow is already saved to the server.
  if (workflow.name.length) {
    // This workflow has a name so let's use the workflow's name with " (copy)" appended to it.
    return `${workflow.name.trim()} (copy)`;
  }
  // Fallback - will show a placeholder in the input field.
  return '';
};

/**
 * Save the workflow as a new workflow. This will open a dialog where the user can enter the name of the new workflow.
 * The workflow object is deep cloned to prevent any changes to the original workflow object.
 * @param workflow The workflow to save as a new workflow.
 */
export const saveWorkflowAs = (workflow: WorkflowV3) => {
  $workflowToSave.set(deepClone(workflow));
};

export const SaveWorkflowAsDialog = () => {
  const isOpen = useStore($isOpen);
  const workflowToSave = useStore($workflowToSave);

  const cancelRef = useRef<HTMLButtonElement>(null);

  const onClose = useCallback(() => {
    $workflowToSave.set(null);
  }, []);

  return (
    <AlertDialog isOpen={isOpen} onClose={onClose} leastDestructiveRef={cancelRef} isCentered={true}>
      {!workflowToSave && <NoWorkflowToSaveContent />}
      {workflowToSave && <Content workflow={workflowToSave} cancelRef={cancelRef} />}
    </AlertDialog>
  );
};

const Content = memo(({ workflow, cancelRef }: { workflow: WorkflowV3; cancelRef: RefObject<HTMLButtonElement> }) => {
  const workflowCategories = useStore($workflowLibraryCategoriesOptions);
  const [name, setName] = useState(() => {
    if (workflow) {
      return getInitialName(workflow);
    }
    return '';
  });
  const [shouldSaveToProject, setShouldSaveToProject] = useState(() => workflowCategories.includes('project'));

  const { createNewWorkflow } = useCreateLibraryWorkflow();

  const inputRef = useRef<HTMLInputElement>(null);

  const onChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setName(e.target.value);
  }, []);

  const onChangeCheckbox = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      setShouldSaveToProject(e.target.checked);
    },
    [setShouldSaveToProject]
  );

  const onClose = useCallback(() => {
    $workflowToSave.set(null);
  }, []);

  const onSave = useCallback(async () => {
    workflow.id = undefined;
    workflow.name = name;
    workflow.meta.category = shouldSaveToProject ? 'project' : 'user';
    workflow.is_published = false;

    // We've just made the workflow a draft, but TS doesn't know that. We need to assert it.
    assert(isDraftWorkflow(workflow));

    await createNewWorkflow({
      workflow,
      onSuccess: onClose,
      onError: onClose,
    });
  }, [workflow, name, shouldSaveToProject, createNewWorkflow, onClose]);

  return (
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
              <Checkbox isChecked={shouldSaveToProject} onChange={onChangeCheckbox}>
                <FormLabel>{t('workflows.saveWorkflowToProject')}</FormLabel>
              </Checkbox>
            )}
          </Flex>
        </FormControl>
      </AlertDialogBody>

      <AlertDialogFooter>
        <Button ref={cancelRef} onClick={onClose}>
          {t('common.cancel')}
        </Button>
        <Button colorScheme="invokeBlue" onClick={onSave} ml={3} isDisabled={!name || !name.length}>
          {t('common.saveAs')}
        </Button>
      </AlertDialogFooter>
    </AlertDialogContent>
  );
});
Content.displayName = 'Content';

const NoWorkflowToSaveContent = memo(() => {
  return (
    <AlertDialogContent>
      <IAINoContentFallback icon={null} label="No workflow to save" />
    </AlertDialogContent>
  );
});
NoWorkflowToSaveContent.displayName = 'NoWorkflowToSaveContent';
