import { Flex, IconButton, Spacer, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ModeToggle } from 'features/nodes/components/sidePanel/ModeToggle';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { $isWorkflowListMenuIsOpen } from 'features/nodes/store/workflowListMenu';
import { selectWorkflowDescription, selectWorkflowName } from 'features/nodes/store/workflowSlice';
import type { MouseEventHandler } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

import SaveWorkflowButton from './SaveWorkflowButton';

export const ActiveWorkflow = () => {
  const activeWorkflowName = useAppSelector(selectWorkflowName);
  const activeWorkflowDescription = useAppSelector(selectWorkflowDescription);

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleNewWorkflow = useCallback<MouseEventHandler<HTMLButtonElement>>(
    (e) => {
      e.stopPropagation();
      dispatch(nodeEditorReset());
      $isWorkflowListMenuIsOpen.set(false);
    },
    [dispatch]
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
      <ModeToggle />
      <SaveWorkflowButton />
      <Tooltip label="Clear workflow">
        <IconButton
          onClick={handleNewWorkflow}
          variant="outline"
          size="sm"
          aria-label="Clear workflow"
          icon={<PiXBold />}
          colorScheme="error"
        />
      </Tooltip>
    </Flex>
  );
};
