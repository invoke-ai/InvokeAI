import {
  Divider,
  Flex,
  FormLabelProps,
  Heading,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
  forwardRef,
  useDisclosure,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIIconButton from 'common/components/IAIIconButton';
import IAISwitch from 'common/components/IAISwitch';
import {
  selectionModeChanged,
  shouldAnimateEdgesChanged,
  shouldColorEdgesChanged,
  shouldSnapToGridChanged,
  shouldValidateGraphChanged,
} from 'features/nodes/store/nodesSlice';
import { ChangeEvent, memo, useCallback } from 'react';
import { FaCog } from 'react-icons/fa';
import { SelectionMode } from 'reactflow';
import ReloadNodeTemplatesButton from '../TopCenterPanel/ReloadSchemaButton';

const formLabelProps: FormLabelProps = {
  fontWeight: 600,
};

const selector = createSelector(
  stateSelector,
  ({ nodes }) => {
    const {
      shouldAnimateEdges,
      shouldValidateGraph,
      shouldSnapToGrid,
      shouldColorEdges,
      selectionMode,
    } = nodes;
    return {
      shouldAnimateEdges,
      shouldValidateGraph,
      shouldSnapToGrid,
      shouldColorEdges,
      selectionModeIsChecked: selectionMode === SelectionMode.Full,
    };
  },
  defaultSelectorOptions
);

const WorkflowEditorSettings = forwardRef((_, ref) => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const dispatch = useAppDispatch();
  const {
    shouldAnimateEdges,
    shouldValidateGraph,
    shouldSnapToGrid,
    shouldColorEdges,
    selectionModeIsChecked,
  } = useAppSelector(selector);

  const handleChangeShouldValidate = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldValidateGraphChanged(e.target.checked));
    },
    [dispatch]
  );

  const handleChangeShouldAnimate = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldAnimateEdgesChanged(e.target.checked));
    },
    [dispatch]
  );

  const handleChangeShouldSnap = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldSnapToGridChanged(e.target.checked));
    },
    [dispatch]
  );

  const handleChangeShouldColor = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldColorEdgesChanged(e.target.checked));
    },
    [dispatch]
  );

  const handleChangeSelectionMode = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(selectionModeChanged(e.target.checked));
    },
    [dispatch]
  );

  return (
    <>
      <IAIIconButton
        ref={ref}
        aria-label="Workflow Editor Settings"
        tooltip="Workflow Editor Settings"
        icon={<FaCog />}
        onClick={onOpen}
      />

      <Modal isOpen={isOpen} onClose={onClose} size="2xl" isCentered>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Workflow Editor Settings</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <Flex
              sx={{
                flexDirection: 'column',
                gap: 4,
                py: 4,
              }}
            >
              <Heading size="sm">General</Heading>
              <IAISwitch
                formLabelProps={formLabelProps}
                onChange={handleChangeShouldAnimate}
                isChecked={shouldAnimateEdges}
                label="Animated Edges"
                helperText="Animate selected edges and edges connected to selected nodes"
              />
              <Divider />
              <IAISwitch
                formLabelProps={formLabelProps}
                isChecked={shouldSnapToGrid}
                onChange={handleChangeShouldSnap}
                label="Snap to Grid"
                helperText="Snap nodes to grid when moved"
              />
              <Divider />
              <IAISwitch
                formLabelProps={formLabelProps}
                isChecked={shouldColorEdges}
                onChange={handleChangeShouldColor}
                label="Color-Code Edges"
                helperText="Color-code edges according to their connected fields"
              />
              <IAISwitch
                formLabelProps={formLabelProps}
                isChecked={selectionModeIsChecked}
                onChange={handleChangeSelectionMode}
                label="Fully Contain Nodes to Select"
                helperText="Nodes must be fully inside the selection box to be selected"
              />
              <Heading size="sm" pt={4}>
                Advanced
              </Heading>
              <IAISwitch
                formLabelProps={formLabelProps}
                isChecked={shouldValidateGraph}
                onChange={handleChangeShouldValidate}
                label="Validate Connections and Graph"
                helperText="Prevent invalid connections from being made, and invalid graphs from being invoked"
              />
              <ReloadNodeTemplatesButton />
            </Flex>
          </ModalBody>
        </ModalContent>
      </Modal>
    </>
  );
});

export default memo(WorkflowEditorSettings);
