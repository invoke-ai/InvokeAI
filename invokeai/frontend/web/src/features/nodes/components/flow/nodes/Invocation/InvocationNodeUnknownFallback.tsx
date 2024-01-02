import { Flex } from '@chakra-ui/react';
import { InvText } from 'common/components/InvText/wrapper';
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

const InvocationNodeUnknownFallback = ({
  nodeId,
  isOpen,
  label,
  type,
  selected,
}: Props) => {
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
        <InvText w="full" textAlign="center" pe={8} color="error.300">
          {label ? `${label} (${type})` : type}
        </InvText>
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
            <InvText as="span">
              {t('nodes.unknownNodeType')}:{' '}
              <InvText as="span" fontWeight="semibold">
                {type}
              </InvText>
            </InvText>
            {nodePack && (
              <InvText as="span">
                {t('nodes.nodePack')}:{' '}
                <InvText as="span" fontWeight="semibold">
                  {nodePack}
                </InvText>
              </InvText>
            )}
          </Flex>
        </Flex>
      )}
    </NodeWrapper>
  );
};

export default memo(InvocationNodeUnknownFallback);
