import { Box, Flex, FormControl, FormLabel, HStack, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { InvocationNodeNotesTextarea } from 'features/nodes/components/flow/nodes/Invocation/InvocationNodeNotesTextarea';
import { TemplateGate } from 'features/nodes/components/sidePanel/inspector/NodeTemplateGate';
import { useNodeNeedsUpdate } from 'features/nodes/hooks/useNodeNeedsUpdate';
import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import { useNodeVersion } from 'features/nodes/hooks/useNodeVersion';
import { selectLastSelectedNodeId } from 'features/nodes/store/selectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import InspectorTabEditableNodeTitle from './InspectorTabEditableNodeTitle';

const InspectorDetailsTab = () => {
  const { t } = useTranslation();
  const lastSelectedNodeId = useAppSelector(selectLastSelectedNodeId);

  if (!lastSelectedNodeId) {
    return <IAINoContentFallback label={t('nodes.noNodeSelected')} icon={null} />;
  }

  return (
    <TemplateGate
      nodeId={lastSelectedNodeId}
      fallback={<IAINoContentFallback label={t('nodes.noNodeSelected')} icon={null} />}
    >
      <Content nodeId={lastSelectedNodeId} />
    </TemplateGate>
  );
};

export default memo(InspectorDetailsTab);

const Content = memo(({ nodeId }: { nodeId: string }) => {
  const { t } = useTranslation();
  const version = useNodeVersion(nodeId);
  const template = useNodeTemplate(nodeId);
  const needsUpdate = useNodeNeedsUpdate(nodeId);

  return (
    <Box position="relative" w="full" h="full">
      <ScrollableContent>
        <Flex flexDir="column" position="relative" w="full" h="full" gap={1}>
          <FormControl>
            <FormLabel m={0}>{t('nodes.nodeName')}</FormLabel>
            <InspectorTabEditableNodeTitle nodeId={nodeId} />
          </FormControl>
          <HStack>
            <FormControl>
              <FormLabel m={0}>{t('nodes.nodeType')}</FormLabel>
              <Text fontSize="sm" fontWeight="semibold">
                {template.title}
              </Text>
            </FormControl>
            <FormControl isInvalid={needsUpdate}>
              <FormLabel m={0}>{t('nodes.nodeVersion')}</FormLabel>
              <Text fontSize="sm" fontWeight="semibold">
                {version}
              </Text>
            </FormControl>
          </HStack>
          <InvocationNodeNotesTextarea nodeId={nodeId} />
        </Flex>
      </ScrollableContent>
    </Box>
  );
});

Content.displayName = 'Content';
