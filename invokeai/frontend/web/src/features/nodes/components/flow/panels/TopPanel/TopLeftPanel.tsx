import { Alert, AlertDescription, AlertIcon, AlertTitle, Box, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import AddNodeButton from 'features/nodes/components/flow/panels/TopPanel/AddNodeButton';
import UpdateNodesButton from 'features/nodes/components/flow/panels/TopPanel/UpdateNodesButton';
import { $isInDeployFlow } from 'features/nodes/components/sidePanel/workflow/publish';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const TopLeftPanel = memo(() => {
  const isInDeployFlow = useStore($isInDeployFlow);

  const { t } = useTranslation();
  return (
    <Flex gap={2} top={2} left={2} position="absolute" alignItems="flex-start" pointerEvents="none">
      {!isInDeployFlow && (
        <Flex gap="2">
          <AddNodeButton />
          <UpdateNodesButton />
        </Flex>
      )}
      {isInDeployFlow && (
        <Alert status="info" borderRadius="base" fontSize="sm" shadow="md" w="fit-content">
          <AlertIcon />
          <Box>
            <AlertTitle>{t('workflows.builder.configuringWorkflowForPublishing')}</AlertTitle>
            <AlertDescription>{t('workflows.builder.configuringWorkflowForPublishingDesc')}</AlertDescription>
          </Box>
        </Alert>
      )}
    </Flex>
  );
});

TopLeftPanel.displayName = 'TopLeftPanel';
