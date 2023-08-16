import {
  Flex,
  FormControl,
  FormLabel,
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
import { useAppDispatch } from 'app/store/storeHooks';
import IAITextarea from 'common/components/IAITextarea';
import {
  useNodeData,
  useNodeLabel,
  useNodeTemplate,
  useNodeTemplateTitle,
} from 'features/nodes/hooks/useNodeData';
import { nodeNotesChanged } from 'features/nodes/store/nodesSlice';
import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import { isInvocationNodeData } from 'features/nodes/types/types';
import { ChangeEvent, memo, useCallback } from 'react';
import { FaInfoCircle } from 'react-icons/fa';

interface Props {
  nodeId: string;
}

const NodeNotesEdit = ({ nodeId }: Props) => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const label = useNodeLabel(nodeId);
  const title = useNodeTemplateTitle(nodeId);

  return (
    <>
      <Tooltip
        label={<TooltipContent nodeId={nodeId} />}
        placement="top"
        shouldWrapChildren
      >
        <Flex
          className={DRAG_HANDLE_CLASSNAME}
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
            sx={{ boxSize: 4, w: 8, color: 'base.400' }}
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

export default memo(NodeNotesEdit);

const TooltipContent = memo(({ nodeId }: { nodeId: string }) => {
  const data = useNodeData(nodeId);
  const nodeTemplate = useNodeTemplate(nodeId);

  if (!isInvocationNodeData(data)) {
    return <Text sx={{ fontWeight: 600 }}>Unknown Node</Text>;
  }

  return (
    <Flex sx={{ flexDir: 'column' }}>
      <Text sx={{ fontWeight: 600 }}>{nodeTemplate?.title}</Text>
      <Text sx={{ opacity: 0.7, fontStyle: 'oblique 5deg' }}>
        {nodeTemplate?.description}
      </Text>
      {data?.notes && <Text>{data.notes}</Text>}
    </Flex>
  );
});

TooltipContent.displayName = 'TooltipContent';

const NotesTextarea = memo(({ nodeId }: { nodeId: string }) => {
  const dispatch = useAppDispatch();
  const data = useNodeData(nodeId);
  const handleNotesChanged = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      dispatch(nodeNotesChanged({ nodeId, notes: e.target.value }));
    },
    [dispatch, nodeId]
  );
  if (!isInvocationNodeData(data)) {
    return null;
  }
  return (
    <FormControl>
      <FormLabel>Notes</FormLabel>
      <IAITextarea
        value={data?.notes}
        onChange={handleNotesChanged}
        rows={10}
      />
    </FormControl>
  );
});

NotesTextarea.displayName = 'NodesTextarea';
