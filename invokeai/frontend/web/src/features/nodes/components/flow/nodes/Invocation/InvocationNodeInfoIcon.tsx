import { Flex, Icon, Text, Tooltip } from '@invoke-ai/ui-library';
import { compare } from 'compare-versions';
import { useNodeData } from 'features/nodes/hooks/useNodeData';
import { useNodeNeedsUpdate } from 'features/nodes/hooks/useNodeNeedsUpdate';
import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import { isInvocationNodeData } from 'features/nodes/types/invocation';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiInfoBold } from 'react-icons/pi';

interface Props {
  nodeId: string;
}

const InvocationNodeInfoIcon = ({ nodeId }: Props) => {
  const needsUpdate = useNodeNeedsUpdate(nodeId);

  return (
    <Tooltip label={<TooltipContent nodeId={nodeId} />} placement="top" shouldWrapChildren>
      <Icon as={PiInfoBold} display="block" boxSize={4} w={8} color={needsUpdate ? 'error.400' : 'base.400'} />
    </Tooltip>
  );
};

export default memo(InvocationNodeInfoIcon);

const TooltipContent = memo(({ nodeId }: { nodeId: string }) => {
  const data = useNodeData(nodeId);
  const nodeTemplate = useNodeTemplate(nodeId);
  const { t } = useTranslation();

  const title = useMemo(() => {
    if (data?.label && nodeTemplate?.title) {
      return `${data.label} (${nodeTemplate.title})`;
    }

    if (data?.label && !nodeTemplate) {
      return data.label;
    }

    if (!data?.label && nodeTemplate) {
      return nodeTemplate.title;
    }

    return t('nodes.unknownNode');
  }, [data, nodeTemplate, t]);

  const versionComponent = useMemo(() => {
    if (!isInvocationNodeData(data) || !nodeTemplate) {
      return null;
    }

    if (!data.version) {
      return (
        <Text as="span" color="error.500">
          {t('nodes.versionUnknown')}
        </Text>
      );
    }

    if (!nodeTemplate.version) {
      return (
        <Text as="span" color="error.500">
          {t('nodes.version')} {data.version} ({t('nodes.unknownTemplate')})
        </Text>
      );
    }

    if (compare(data.version, nodeTemplate.version, '<')) {
      return (
        <Text as="span" color="error.500">
          {t('nodes.version')} {data.version} ({t('nodes.updateNode')})
        </Text>
      );
    }

    if (compare(data.version, nodeTemplate.version, '>')) {
      return (
        <Text as="span" color="error.500">
          {t('nodes.version')} {data.version} ({t('nodes.updateApp')})
        </Text>
      );
    }

    return (
      <Text as="span">
        {t('nodes.version')} {data.version}
      </Text>
    );
  }, [data, nodeTemplate, t]);

  if (!isInvocationNodeData(data)) {
    return <Text fontWeight="semibold">{t('nodes.unknownNode')}</Text>;
  }

  return (
    <Flex flexDir="column">
      <Text as="span" fontWeight="semibold">
        {title}
      </Text>
      {nodeTemplate?.nodePack && (
        <Text opacity={0.7}>
          {t('nodes.nodePack')}: {nodeTemplate.nodePack}
        </Text>
      )}
      <Text opacity={0.7} fontStyle="oblique 5deg">
        {nodeTemplate?.description}
      </Text>
      {versionComponent}
      {data?.notes && <Text>{data.notes}</Text>}
    </Flex>
  );
});

TooltipContent.displayName = 'TooltipContent';
