import { Flex, IconButton, Spacer, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { ModeToggle } from 'features/nodes/components/sidePanel/ModeToggle';
import { selectWorkflowDescription, selectWorkflowMode, selectWorkflowName } from 'features/nodes/store/workflowSlice';
import { useNewWorkflow } from 'features/workflowLibrary/components/NewWorkflowConfirmationAlertDialog';
import type { MouseEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFilePlusBold } from 'react-icons/pi';

import SaveWorkflowButton from './SaveWorkflowButton';

export const ActiveWorkflow = () => {
  const activeWorkflowName = useAppSelector(selectWorkflowName);
  const activeWorkflowDescription = useAppSelector(selectWorkflowDescription);
  const mode = useAppSelector(selectWorkflowMode);
  const newWorkflow = useNewWorkflow();

  const { t } = useTranslation();

  const onClickNewWorkflow = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      // We need to stop the event from propagating to the parent element, else the click will open the list menu
      e.stopPropagation();
      newWorkflow.createWithDialog();
    },
    [newWorkflow]
  );

  return (
    <Flex w="full" alignItems="center" gap={2} minW={0}>
      {activeWorkflowName ? (
        <Tooltip label={activeWorkflowDescription}>
          <Text colorScheme="invokeBlue" fontWeight="semibold" fontSize="md" justifySelf="flex-start">
            {activeWorkflowName}
          </Text>
        </Tooltip>
      ) : (
        <Text fontSize="sm" fontWeight="semibold" color="base.300">
          {t('workflows.chooseWorkflowFromLibrary')}
        </Text>
      )}
      <Spacer />
      {mode === 'edit' && <SaveWorkflowButton />}
      <ModeToggle />
      <IconButton
        onClick={onClickNewWorkflow}
        variant="outline"
        size="sm"
        aria-label={t('nodes.newWorkflow')}
        tooltip={t('nodes.newWorkflow')}
        icon={<PiFilePlusBold />}
      />
    </Flex>
  );
};
