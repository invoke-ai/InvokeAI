import { Flex, Icon, Text, Tooltip } from '@chakra-ui/react';
import { compare } from 'compare-versions';
import { useNodeData } from 'features/nodes/hooks/useNodeData';
import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import { useNodeNeedsUpdate } from 'features/nodes/hooks/useNodeNeedsUpdate';
import { isInvocationNodeData } from 'features/nodes/types/invocation';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaInfoCircle } from 'react-icons/fa';

interface Props {
  nodeId: string;
}

const InvocationNodeInfoIcon = ({ nodeId }: Props) => {
  const needsUpdate = useNodeNeedsUpdate(nodeId);

  return (
    <Tooltip
      label={<TooltipContent nodeId={nodeId} />}
      placement="top"
      shouldWrapChildren
    >
      <Icon
        as={FaInfoCircle}
        sx={{
          boxSize: 4,
          w: 8,
          color: needsUpdate ? 'error.400' : 'base.400',
        }}
      />
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
        <Text as="span" sx={{ color: 'error.500' }}>
          {t('nodes.versionUnknown')}
        </Text>
      );
    }

    if (!nodeTemplate.version) {
      return (
        <Text as="span" sx={{ color: 'error.500' }}>
          {t('nodes.version')} {data.version} ({t('nodes.unknownTemplate')})
        </Text>
      );
    }

    if (compare(data.version, nodeTemplate.version, '<')) {
      return (
        <Text as="span" sx={{ color: 'error.500' }}>
          {t('nodes.version')} {data.version} ({t('nodes.updateNode')})
        </Text>
      );
    }

    if (compare(data.version, nodeTemplate.version, '>')) {
      return (
        <Text as="span" sx={{ color: 'error.500' }}>
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
    return <Text sx={{ fontWeight: 600 }}>{t('nodes.unknownNode')}</Text>;
  }

  return (
    <Flex sx={{ flexDir: 'column' }}>
      <Text as="span" sx={{ fontWeight: 600 }}>
        {title}
      </Text>
      <Text sx={{ opacity: 0.7, fontStyle: 'oblique 5deg' }}>
        {nodeTemplate?.description}
      </Text>
      {versionComponent}
      {data?.notes && <Text>{data.notes}</Text>}
    </Flex>
  );
});

TooltipContent.displayName = 'TooltipContent';
