import { Divider, Flex, Heading, useDisclosure } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import {
  InvModal,
  InvModalBody,
  InvModalCloseButton,
  InvModalContent,
  InvModalHeader,
  InvModalOverlay,
} from 'common/components/InvModal/wrapper';
import { InvSwitch } from 'common/components/InvSwitch/wrapper';
import ReloadNodeTemplatesButton from 'features/nodes/components/flow/panels/TopRightPanel/ReloadSchemaButton';
import {
  selectionModeChanged,
  shouldAnimateEdgesChanged,
  shouldColorEdgesChanged,
  shouldSnapToGridChanged,
  shouldValidateGraphChanged,
} from 'features/nodes/store/nodesSlice';
import type { ChangeEvent, ReactNode } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { SelectionMode } from 'reactflow';

const selector = createMemoizedSelector(stateSelector, ({ nodes }) => {
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
});

type Props = {
  children: (props: { onOpen: () => void }) => ReactNode;
};

const WorkflowEditorSettings = ({ children }: Props) => {
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
      {children({ onOpen })}

      <InvModal isOpen={isOpen} onClose={onClose} size="2xl" isCentered>
        <InvModalOverlay />
        <InvModalContent>
          <InvModalHeader>{t('nodes.workflowSettings')}</InvModalHeader>
          <InvModalCloseButton />
          <InvModalBody>
            <Flex
              sx={{
                flexDirection: 'column',
                gap: 4,
                py: 4,
              }}
            >
              <Heading size="sm">{t('parameters.general')}</Heading>
              <InvControl
                label={t('nodes.animatedEdges')}
                helperText={t('nodes.animatedEdgesHelp')}
              >
                <InvSwitch
                  onChange={handleChangeShouldAnimate}
                  isChecked={shouldAnimateEdges}
                />
              </InvControl>
              <Divider />
              <InvControl
                label={t('nodes.snapToGrid')}
                helperText={t('nodes.snapToGridHelp')}
              >
                <InvSwitch
                  isChecked={shouldSnapToGrid}
                  onChange={handleChangeShouldSnap}
                />
              </InvControl>
              <Divider />
              <InvControl
                label={t('nodes.colorCodeEdges')}
                helperText={t('nodes.colorCodeEdgesHelp')}
              >
                <InvSwitch
                  isChecked={shouldColorEdges}
                  onChange={handleChangeShouldColor}
                />
              </InvControl>
              <Divider />
              <InvControl
                label={t('nodes.fullyContainNodes')}
                helperText={t('nodes.fullyContainNodesHelp')}
              >
                <InvSwitch
                  isChecked={selectionModeIsChecked}
                  onChange={handleChangeSelectionMode}
                />
              </InvControl>
              <Heading size="sm" pt={4}>
                {t('common.advanced')}
              </Heading>
              <InvControl
                label={t('nodes.validateConnections')}
                helperText={t('nodes.validateConnectionsHelp')}
              >
                <InvSwitch
                  isChecked={shouldValidateGraph}
                  onChange={handleChangeShouldValidate}
                />
              </InvControl>
              <ReloadNodeTemplatesButton />
            </Flex>
          </InvModalBody>
        </InvModalContent>
      </InvModal>
    </>
  );
};

export default memo(WorkflowEditorSettings);
