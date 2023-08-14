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
import { nodeNotesChanged } from 'features/nodes/store/nodesSlice';
import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import {
  InvocationNodeData,
  InvocationTemplate,
} from 'features/nodes/types/types';
import { ChangeEvent, memo, useCallback } from 'react';
import { FaInfoCircle } from 'react-icons/fa';
import { NodeProps } from 'reactflow';

interface Props {
  nodeProps: NodeProps<InvocationNodeData>;
  nodeTemplate: InvocationTemplate;
}

const NodeNotesEdit = (props: Props) => {
  const { nodeProps, nodeTemplate } = props;
  const { data } = nodeProps;
  const { isOpen, onOpen, onClose } = useDisclosure();
  const dispatch = useAppDispatch();
  const handleNotesChanged = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      dispatch(nodeNotesChanged({ nodeId: data.id, notes: e.target.value }));
    },
    [data.id, dispatch]
  );

  return (
    <>
      <Tooltip
        label={
          nodeTemplate ? (
            <TooltipContent nodeProps={nodeProps} nodeTemplate={nodeTemplate} />
          ) : undefined
        }
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
          <ModalHeader>
            {data.label || nodeTemplate?.title || 'Unknown Node'}
          </ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <FormControl>
              <FormLabel>Notes</FormLabel>
              <IAITextarea
                value={data.notes}
                onChange={handleNotesChanged}
                rows={10}
              />
            </FormControl>
          </ModalBody>
          <ModalFooter />
        </ModalContent>
      </Modal>
    </>
  );
};

export default memo(NodeNotesEdit);

type TooltipContentProps = Props;

const TooltipContent = (props: TooltipContentProps) => {
  return (
    <Flex sx={{ flexDir: 'column' }}>
      <Text sx={{ fontWeight: 600 }}>{props.nodeTemplate?.title}</Text>
      <Text sx={{ opacity: 0.7, fontStyle: 'oblique 5deg' }}>
        {props.nodeTemplate?.description}
      </Text>
      {props.nodeProps.data.notes && <Text>{props.nodeProps.data.notes}</Text>}
    </Flex>
  );
};
