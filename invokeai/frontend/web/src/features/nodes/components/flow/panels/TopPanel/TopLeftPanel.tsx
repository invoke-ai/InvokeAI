import { Alert, AlertDescription, AlertIcon, AlertTitle, Box, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import AddNodeButton from 'features/nodes/components/flow/panels/TopPanel/AddNodeButton';
import UpdateNodesButton from 'features/nodes/components/flow/panels/TopPanel/UpdateNodesButton';
import { $isInPublishFlow, useIsValidationRunInProgress } from 'features/nodes/components/sidePanel/workflow/publish';
import { useIsWorkflowEditorLocked } from 'features/nodes/hooks/useIsWorkflowEditorLocked';
import { selectWorkflowIsPublished } from 'features/nodes/store/workflowSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const TopLeftPanel = memo(() => {
  const isLocked = useIsWorkflowEditorLocked();
  const isInPublishFlow = useStore($isInPublishFlow);
  const isPublished = useAppSelector(selectWorkflowIsPublished);
  const isValidationRunInProgress = useIsValidationRunInProgress();

  const { t } = useTranslation();
  return (
    <Flex gap={2} top={2} left={2} position="absolute" alignItems="flex-start" pointerEvents="none">
      {!isLocked && (
        <Flex gap="2">
          <AddNodeButton />
          <UpdateNodesButton />
        </Flex>
      )}
      {isLocked && (
        <Alert status="info" borderRadius="base" fontSize="sm" shadow="md" w="fit-content">
          <AlertIcon />
          <Box>
            <AlertTitle>{t('workflows.builder.workflowLocked')}</AlertTitle>
            {isValidationRunInProgress && (
              <AlertDescription whiteSpace="pre-wrap">
                {t('workflows.builder.publishingValidationRunInProgress')}
              </AlertDescription>
            )}
            {isInPublishFlow && !isValidationRunInProgress && (
              <AlertDescription whiteSpace="pre-wrap">
                {t('workflows.builder.workflowLockedDuringPublishing')}
              </AlertDescription>
            )}
            {isPublished && (
              <AlertDescription whiteSpace="pre-wrap">
                {t('workflows.builder.workflowLockedPublished')}
              </AlertDescription>
            )}
          </Box>
        </Alert>
      )}
    </Flex>
  );
});

TopLeftPanel.displayName = 'TopLeftPanel';
