import { Flex, Text } from '@invoke-ai/ui-library';
import NodeCollapseButton from 'features/nodes/components/flow/nodes/common/NodeCollapseButton';
import NodeWrapper from 'features/nodes/components/flow/nodes/common/NodeWrapper';
import { useNodePack } from 'features/nodes/hooks/useNodePack';
import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  nodeId: string;
  isOpen: boolean;
  label: string;
  type: string;
  selected: boolean;
};

const InvocationNodeUnknownFallback = ({ nodeId, isOpen, label, type, selected }: Props) => {
  const { t } = useTranslation();
  const nodePack = useNodePack(nodeId);
  return (
    <NodeWrapper nodeId={nodeId} selected={selected}>
      <Flex
        className={DRAG_HANDLE_CLASSNAME}
        layerStyle="nodeHeader"
        borderTopRadius="base"
        borderBottomRadius={isOpen ? 0 : 'base'}
        alignItems="center"
        h={8}
        fontWeight="semibold"
        fontSize="sm"
      >
        <NodeCollapseButton nodeId={nodeId} isOpen={isOpen} />
        <Text w="full" textAlign="center" pe={8} color="error.300">
          {label ? `${label} (${type})` : type}
        </Text>
      </Flex>
      {isOpen && (
        <Flex
          layerStyle="nodeBody"
          userSelect="auto"
          flexDirection="column"
          w="full"
          h="full"
          p={4}
          gap={1}
          borderBottomRadius="base"
          fontSize="sm"
        >
          <Flex gap={2} flexDir="column">
            <Text as="span">
              {t('nodes.unknownNodeType')}:{' '}
              <Text as="span" fontWeight="semibold">
                {type}
              </Text>
            </Text>
            {nodePack && (
              <Text as="span">
                {t('nodes.nodePack')}:{' '}
                <Text as="span" fontWeight="semibold">
                  {nodePack}
                </Text>
              </Text>
            )}
          </Flex>
        </Flex>
      )}
    </NodeWrapper>
  );
};

export default memo(InvocationNodeUnknownFallback);
