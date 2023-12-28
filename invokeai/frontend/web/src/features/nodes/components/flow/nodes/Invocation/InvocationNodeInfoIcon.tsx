import { Flex, Icon } from '@chakra-ui/react';
import { InvText } from 'common/components/InvText/wrapper';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { compare } from 'compare-versions';
import { useNodeData } from 'features/nodes/hooks/useNodeData';
import { useNodeNeedsUpdate } from 'features/nodes/hooks/useNodeNeedsUpdate';
import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
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
    <InvTooltip
      label={<TooltipContent nodeId={nodeId} />}
      placement="top"
      shouldWrapChildren
    >
      <Icon
        as={FaInfoCircle}
        sx={{
          display: 'block',
          boxSize: 4,
          w: 8,
          color: needsUpdate ? 'error.400' : 'base.400',
        }}
      />
    </InvTooltip>
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
        <InvText as="span" sx={{ color: 'error.500' }}>
          {t('nodes.versionUnknown')}
        </InvText>
      );
    }

    if (!nodeTemplate.version) {
      return (
        <InvText as="span" sx={{ color: 'error.500' }}>
          {t('nodes.version')} {data.version} ({t('nodes.unknownTemplate')})
        </InvText>
      );
    }

    if (compare(data.version, nodeTemplate.version, '<')) {
      return (
        <InvText as="span" sx={{ color: 'error.500' }}>
          {t('nodes.version')} {data.version} ({t('nodes.updateNode')})
        </InvText>
      );
    }

    if (compare(data.version, nodeTemplate.version, '>')) {
      return (
        <InvText as="span" sx={{ color: 'error.500' }}>
          {t('nodes.version')} {data.version} ({t('nodes.updateApp')})
        </InvText>
      );
    }

    return (
      <InvText as="span">
        {t('nodes.version')} {data.version}
      </InvText>
    );
  }, [data, nodeTemplate, t]);

  if (!isInvocationNodeData(data)) {
    return (
      <InvText sx={{ fontWeight: 'semibold' }}>
        {t('nodes.unknownNode')}
      </InvText>
    );
  }

  return (
    <Flex sx={{ flexDir: 'column' }}>
      <InvText as="span" sx={{ fontWeight: 'semibold' }}>
        {title}
      </InvText>
      {nodeTemplate?.nodePack && (
        <InvText opacity={0.7}>
          {t('nodes.nodePack')}: {nodeTemplate.nodePack}
        </InvText>
      )}
      <InvText sx={{ opacity: 0.7, fontStyle: 'oblique 5deg' }}>
        {nodeTemplate?.description}
      </InvText>
      {versionComponent}
      {data?.notes && <InvText>{data.notes}</InvText>}
    </Flex>
  );
});

TooltipContent.displayName = 'TooltipContent';
