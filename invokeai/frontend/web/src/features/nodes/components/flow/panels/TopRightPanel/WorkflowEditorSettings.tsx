import type { FormLabelProps } from '@invoke-ai/ui-library';
import {
  Divider,
  Flex,
  FormControl,
  FormControlGroup,
  FormHelperText,
  FormLabel,
  Heading,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
  Switch,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import ReloadNodeTemplatesButton from 'features/nodes/components/flow/panels/TopRightPanel/ReloadSchemaButton';
import {
  selectionModeChanged,
  selectSelectionMode,
  selectShouldAnimateEdges,
  selectShouldColorEdges,
  selectShouldShouldValidateGraph,
  selectShouldShowEdgeLabels,
  selectShouldSnapToGrid,
  shouldAnimateEdgesChanged,
  shouldColorEdgesChanged,
  shouldShowEdgeLabelsChanged,
  shouldSnapToGridChanged,
  shouldValidateGraphChanged,
} from 'features/nodes/store/workflowSettingsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { SelectionMode } from 'reactflow';

const formLabelProps: FormLabelProps = { flexGrow: 1 };
export const [useWorkflowEditorSettingsModal] = buildUseBoolean(false);

const WorkflowEditorSettings = () => {
  const dispatch = useAppDispatch();
  const modal = useWorkflowEditorSettingsModal();

  const shouldSnapToGrid = useAppSelector(selectShouldSnapToGrid);
  const selectionMode = useAppSelector(selectSelectionMode);
  const shouldColorEdges = useAppSelector(selectShouldColorEdges);
  const shouldAnimateEdges = useAppSelector(selectShouldAnimateEdges);
  const shouldShowEdgeLabels = useAppSelector(selectShouldShowEdgeLabels);
  const shouldValidateGraph = useAppSelector(selectShouldShouldValidateGraph);

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

  const handleChangeShouldShowEdgeLabels = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldShowEdgeLabelsChanged(e.target.checked));
    },
    [dispatch]
  );

  const { t } = useTranslation();

  return (
    <Modal isOpen={modal.isTrue} onClose={modal.setFalse} size="2xl" isCentered useInert={false}>
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>{t('nodes.workflowSettings')}</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <Flex flexDirection="column" gap={4} py={4}>
            <Heading size="sm">{t('parameters.general')}</Heading>
            <FormControlGroup orientation="vertical" formLabelProps={formLabelProps}>
              <FormControl>
                <Flex w="full">
                  <FormLabel>{t('nodes.animatedEdges')}</FormLabel>
                  <Switch onChange={handleChangeShouldAnimate} isChecked={shouldAnimateEdges} />
                </Flex>
                <FormHelperText>{t('nodes.animatedEdgesHelp')}</FormHelperText>
              </FormControl>
              <Divider />
              <FormControl>
                <Flex w="full">
                  <FormLabel>{t('nodes.snapToGrid')}</FormLabel>
                  <Switch isChecked={shouldSnapToGrid} onChange={handleChangeShouldSnap} />
                </Flex>
                <FormHelperText>{t('nodes.snapToGridHelp')}</FormHelperText>
              </FormControl>
              <Divider />
              <FormControl>
                <Flex w="full">
                  <FormLabel>{t('nodes.colorCodeEdges')}</FormLabel>
                  <Switch isChecked={shouldColorEdges} onChange={handleChangeShouldColor} />
                </Flex>
                <FormHelperText>{t('nodes.colorCodeEdgesHelp')}</FormHelperText>
              </FormControl>
              <Divider />
              <FormControl>
                <Flex w="full">
                  <FormLabel>{t('nodes.fullyContainNodes')}</FormLabel>
                  <Switch isChecked={selectionMode === SelectionMode.Full} onChange={handleChangeSelectionMode} />
                </Flex>
                <FormHelperText>{t('nodes.fullyContainNodesHelp')}</FormHelperText>
              </FormControl>
              <Divider />
              <FormControl>
                <Flex w="full">
                  <FormLabel>{t('nodes.showEdgeLabels')}</FormLabel>
                  <Switch isChecked={shouldShowEdgeLabels} onChange={handleChangeShouldShowEdgeLabels} />
                </Flex>
                <FormHelperText>{t('nodes.showEdgeLabelsHelp')}</FormHelperText>
              </FormControl>
              <Divider />
              <Heading size="sm" pt={4}>
                {t('common.advanced')}
              </Heading>
              <FormControl>
                <Flex w="full">
                  <FormLabel>{t('nodes.validateConnections')}</FormLabel>
                  <Switch isChecked={shouldValidateGraph} onChange={handleChangeShouldValidate} />
                </Flex>
                <FormHelperText>{t('nodes.validateConnectionsHelp')}</FormHelperText>
              </FormControl>
              <Divider />
            </FormControlGroup>
            <ReloadNodeTemplatesButton />
          </Flex>
        </ModalBody>
      </ModalContent>
    </Modal>
  );
};

export default memo(WorkflowEditorSettings);
