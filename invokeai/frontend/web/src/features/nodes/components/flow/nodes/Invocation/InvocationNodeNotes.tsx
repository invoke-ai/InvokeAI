import {
  Flex,
  Icon,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Text,
  Tooltip,
  useDisclosure,
} from '@chakra-ui/react';
import { compare } from 'compare-versions';
import { useNodeData } from 'features/nodes/hooks/useNodeData';
import { useNodeLabel } from 'features/nodes/hooks/useNodeLabel';
import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import { useNodeTemplateTitle } from 'features/nodes/hooks/useNodeTemplateTitle';
import { isInvocationNodeData } from 'features/nodes/types/types';
import { memo, useMemo } from 'react';
import { FaInfoCircle } from 'react-icons/fa';
import NotesTextarea from './NotesTextarea';
import { useDoNodeVersionsMatch } from 'features/nodes/hooks/useDoNodeVersionsMatch';

interface Props {
  nodeId: string;
}

const InvocationNodeNotes = ({ nodeId }: Props) => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const label = useNodeLabel(nodeId);
  const title = useNodeTemplateTitle(nodeId);
  const doVersionsMatch = useDoNodeVersionsMatch(nodeId);

  return (
    <>
      <Tooltip
        label={<TooltipContent nodeId={nodeId} />}
        placement="top"
        shouldWrapChildren
      >
        <Flex
          className="nodrag"
          onClick={onOpen}
          sx={{
            alignItems: 'center',
            justifyContent: 'center',
            w: 8,
            h: 8,
            cursor: 'pointer',
          }}
        >
          <Icon
            as={FaInfoCircle}
            sx={{
              boxSize: 4,
              w: 8,
              color: doVersionsMatch ? 'base.400' : 'error.400',
            }}
          />
        </Flex>
      </Tooltip>

      <Modal isOpen={isOpen} onClose={onClose} isCentered>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>{label || title || 'Unknown Node'}</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <NotesTextarea nodeId={nodeId} />
          </ModalBody>
          <ModalFooter />
        </ModalContent>
      </Modal>
    </>
  );
};

export default memo(InvocationNodeNotes);

const TooltipContent = memo(({ nodeId }: { nodeId: string }) => {
  const data = useNodeData(nodeId);
  const nodeTemplate = useNodeTemplate(nodeId);

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

    return 'Unknown Node';
  }, [data, nodeTemplate]);

  const versionComponent = useMemo(() => {
    if (!isInvocationNodeData(data) || !nodeTemplate) {
      return null;
    }

    if (!data.version) {
      return (
        <Text as="span" sx={{ color: 'error.500' }}>
          Version unknown
        </Text>
      );
    }

    if (!nodeTemplate.version) {
      return (
        <Text as="span" sx={{ color: 'error.500' }}>
          Version {data.version} (unknown template)
        </Text>
      );
    }

    if (compare(data.version, nodeTemplate.version, '<')) {
      return (
        <Text as="span" sx={{ color: 'error.500' }}>
          Version {data.version} (update node)
        </Text>
      );
    }

    if (compare(data.version, nodeTemplate.version, '>')) {
      return (
        <Text as="span" sx={{ color: 'error.500' }}>
          Version {data.version} (update app)
        </Text>
      );
    }

    return <Text as="span">Version {data.version}</Text>;
  }, [data, nodeTemplate]);

  if (!isInvocationNodeData(data)) {
    return <Text sx={{ fontWeight: 600 }}>Unknown Node</Text>;
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
