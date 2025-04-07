import { Alert, AlertDescription, AlertIcon, AlertTitle, Box, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import AddNodeButton from 'features/nodes/components/flow/panels/TopPanel/AddNodeButton';
import UpdateNodesButton from 'features/nodes/components/flow/panels/TopPanel/UpdateNodesButton';
import {
  $isInPublishFlow,
  $isSelectingOutputNode,
  useIsValidationRunInProgress,
  useIsWorkflowPublished,
} from 'features/nodes/components/sidePanel/workflow/publish';
import { useIsWorkflowEditorLocked } from 'features/nodes/hooks/useIsWorkflowEditorLocked';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const TopLeftPanel = memo(() => {
  const isLocked = useIsWorkflowEditorLocked();
  const isInPublishFlow = useStore($isInPublishFlow);
  const isPublished = useIsWorkflowPublished();
  const isValidationRunInProgress = useIsValidationRunInProgress();
  const isSelectingOutputNode = useStore($isSelectingOutputNode);

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
            {isInPublishFlow && !isValidationRunInProgress && !isSelectingOutputNode && (
              <AlertDescription whiteSpace="pre-wrap">
                {t('workflows.builder.workflowLockedDuringPublishing')}
              </AlertDescription>
            )}
            {isInPublishFlow && !isValidationRunInProgress && isSelectingOutputNode && (
              <AlertDescription whiteSpace="pre-wrap">
                {t('workflows.builder.selectingOutputNodeDesc')}
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
