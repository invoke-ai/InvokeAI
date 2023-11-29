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
import ReloadNodeTemplatesButton from 'features/nodes/components/flow/panels/TopCenterPanel/ReloadSchemaButton';
import { useTranslation } from 'react-i18next';

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

  const { t } = useTranslation();

  return (
    <>
      <IAIIconButton
        ref={ref}
        aria-label={t('nodes.workflowSettings')}
        tooltip={t('nodes.workflowSettings')}
        icon={<FaCog />}
        onClick={onOpen}
      />

      <Modal isOpen={isOpen} onClose={onClose} size="2xl" isCentered>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>{t('nodes.workflowSettings')}</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <Flex
              sx={{
                flexDirection: 'column',
                gap: 4,
                py: 4,
              }}
            >
              <Heading size="sm">{t('parameters.general')}</Heading>
              <IAISwitch
                formLabelProps={formLabelProps}
                onChange={handleChangeShouldAnimate}
                isChecked={shouldAnimateEdges}
                label={t('nodes.animatedEdges')}
                helperText={t('nodes.animatedEdgesHelp')}
              />
              <Divider />
              <IAISwitch
                formLabelProps={formLabelProps}
                isChecked={shouldSnapToGrid}
                onChange={handleChangeShouldSnap}
                label={t('nodes.snapToGrid')}
                helperText={t('nodes.snapToGridHelp')}
              />
              <Divider />
              <IAISwitch
                formLabelProps={formLabelProps}
                isChecked={shouldColorEdges}
                onChange={handleChangeShouldColor}
                label={t('nodes.colorCodeEdges')}
                helperText={t('nodes.colorCodeEdgesHelp')}
              />
              <IAISwitch
                formLabelProps={formLabelProps}
                isChecked={selectionModeIsChecked}
                onChange={handleChangeSelectionMode}
                label={t('nodes.fullyContainNodes')}
                helperText={t('nodes.fullyContainNodesHelp')}
              />
              <Heading size="sm" pt={4}>
                {t('common.advanced')}
              </Heading>
              <IAISwitch
                formLabelProps={formLabelProps}
                isChecked={shouldValidateGraph}
                onChange={handleChangeShouldValidate}
                label={t('nodes.validateConnections')}
                helperText={t('nodes.validateConnectionsHelp')}
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
