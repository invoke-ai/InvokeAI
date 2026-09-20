import {
  Button,
  ButtonGroup,
  ConfirmationAlertDialog,
  Flex,
  FormControl,
  FormLabel,
  Heading,
  Kbd,
  Radio,
  RadioGroup,
  Text,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityTypeIsHidden } from 'features/controlLayers/hooks/useEntityTypeIsHidden';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { BezierPointType } from 'features/controlLayers/util/bezierPath';
import { canJoinVectorPathEndpoints, canSplitVectorPathAtPoints } from 'features/controlLayers/util/vectorPathTopology';
import { useHotkeyData } from 'features/system/components/HotkeysModal/useHotkeyData';
import { Fragment, memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

const BEZIER_POINT_TYPES = ['corner', 'smooth', 'symmetric'] as const satisfies readonly BezierPointType[];

const isBezierPointType = (value: string): value is BezierPointType => {
  return BEZIER_POINT_TYPES.includes(value as BezierPointType);
};

export const VectorLayerEditFooter = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const hotkeysData = useHotkeyData();
  const [smoothTarget, setSmoothTarget] = useState<'path' | 'selected'>('path');
  const editSession = useStore(canvasManager.tool.tools.path.$editSession);
  const isExitConfirmationOpen = useStore(canvasManager.tool.tools.path.$isExitConfirmationOpen);
  const transformingAdapter = useStore(canvasManager.stateApi.$transformingAdapter);
  const isBusy = useCanvasIsBusy();
  const isVectorLayerTypeHidden = useEntityTypeIsHidden('vector_layer');
  const isEditSessionEntityMutable = useAppSelector((state) => {
    if (!editSession) {
      return false;
    }

    const canvas = selectCanvasSlice(state);
    const selectedEntityIdentifier = canvas.selectedEntityIdentifier;
    if (
      !selectedEntityIdentifier ||
      selectedEntityIdentifier.id !== editSession.entityIdentifier.id ||
      selectedEntityIdentifier.type !== editSession.entityIdentifier.type
    ) {
      return false;
    }

    const layer = canvas.vectorLayers.entities.find((entity) => entity.id === editSession.entityIdentifier.id);
    return Boolean(layer?.isEnabled && !layer.isLocked);
  });
  const selectedPointType = useAppSelector((state): BezierPointType | 'mixed' | null => {
    if (!editSession || !editSession.activePathId || editSession.activePointIndex === null) {
      return null;
    }

    const canvas = selectCanvasSlice(state);
    const layer = canvas.vectorLayers.entities.find((entity) => entity.id === editSession.entityIdentifier.id);
    const selectedPoints =
      editSession.selectedPoints.length > 0
        ? editSession.selectedPoints
        : [{ pathId: editSession.activePathId, pointIndex: editSession.activePointIndex }];
    const pointTypes = selectedPoints.flatMap((pointRef) => {
      const path = layer?.paths.find((candidate) => candidate.id === pointRef.pathId);
      const point = path?.points[pointRef.pointIndex];
      return point ? [point.type] : [];
    });
    const firstPointType = pointTypes[0];
    if (!firstPointType) {
      return null;
    }
    return pointTypes.every((pointType) => pointType === firstPointType) ? firstPointType : 'mixed';
  });
  const canSmoothActivePath = useAppSelector((state) => {
    if (!editSession?.activePathId) {
      return false;
    }

    const canvas = selectCanvasSlice(state);
    const layer = canvas.vectorLayers.entities.find((entity) => entity.id === editSession.entityIdentifier.id);
    const path = layer?.paths.find((path) => path.id === editSession.activePathId);
    return Boolean(path && path.points.length >= 2);
  });
  const canSmoothSelectedPoints = useAppSelector((state) => {
    if (!editSession || editSession.selectedPoints.length === 0) {
      return false;
    }

    const canvas = selectCanvasSlice(state);
    const layer = canvas.vectorLayers.entities.find((entity) => entity.id === editSession.entityIdentifier.id);
    return Boolean(
      layer &&
      editSession.selectedPoints.some((pointRef) => {
        const path = layer.paths.find((candidate) => candidate.id === pointRef.pathId);
        return Boolean(path && path.points.length >= 2 && path.points[pointRef.pointIndex]);
      })
    );
  });
  const editSessionPaths = useAppSelector((state) => {
    if (!editSession) {
      return [];
    }
    const canvas = selectCanvasSlice(state);
    return canvas.vectorLayers.entities.find((entity) => entity.id === editSession.entityIdentifier.id)?.paths ?? [];
  });

  const onPointTypeChange = useCallback(
    (value: string) => {
      if (!isBezierPointType(value)) {
        return;
      }
      canvasManager.tool.tools.path.setActivePointType(value);
    },
    [canvasManager.tool.tools.path]
  );
  const onApply = useCallback(() => {
    canvasManager.tool.tools.path.acceptEditSession();
  }, [canvasManager.tool.tools.path]);
  const onReset = useCallback(() => {
    canvasManager.tool.tools.path.resetEditSession();
    setSmoothTarget('path');
  }, [canvasManager.tool.tools.path]);
  const onSmoothPath = useCallback(() => {
    if (smoothTarget === 'selected') {
      canvasManager.tool.tools.path.smoothSelectedPoints();
    } else {
      canvasManager.tool.tools.path.smoothActivePath();
    }
  }, [canvasManager.tool.tools.path, smoothTarget]);
  const onSmoothSelectedPointsModeToggle = useCallback(() => {
    setSmoothTarget((currentTarget) => (currentTarget === 'selected' ? 'path' : 'selected'));
  }, []);
  const onSplitPoint = useCallback(() => {
    canvasManager.tool.tools.path.splitSelectedPoints();
  }, [canvasManager.tool.tools.path]);
  const onJoinPoints = useCallback(() => {
    canvasManager.tool.tools.path.joinSelectedEndpoints();
  }, [canvasManager.tool.tools.path]);
  const onApplyToolChange = useCallback(() => {
    canvasManager.tool.tools.path.acceptEditSession(false);
  }, [canvasManager.tool.tools.path]);
  const onDiscardToolChange = useCallback(() => {
    canvasManager.tool.tools.path.discardEditSession(false);
  }, [canvasManager.tool.tools.path]);
  const onCancelToolChange = useCallback(() => {
    canvasManager.tool.tools.path.cancelToolChange();
  }, [canvasManager.tool.tools.path]);

  if (!editSession || transformingAdapter) {
    return null;
  }

  const canMutateEditSession = isEditSessionEntityMutable && !isBusy && !isVectorLayerTypeHidden;
  const selectedPointRefs =
    editSession.selectedPoints.length > 0
      ? editSession.selectedPoints
      : editSession.activePathId && editSession.activePointIndex !== null
        ? [{ pathId: editSession.activePathId, pointIndex: editSession.activePointIndex }]
        : [];
  const canSplitPoint = canMutateEditSession && canSplitVectorPathAtPoints(editSessionPaths, selectedPointRefs);
  const canJoinPoints =
    canMutateEditSession &&
    canJoinVectorPathEndpoints(
      editSessionPaths,
      editSession.selectedPoints,
      canvasManager.stage.unscale(canvasManager.tool.tools.path.config.JOIN_POINT_WELD_RADIUS_PX)
    );
  const deletePathKeys = hotkeysData.canvas.hotkeys.deleteSelected?.platformKeys[0] ?? ['Delete'];

  return (
    <Fragment>
      <ConfirmationAlertDialog
        isOpen={isExitConfirmationOpen}
        onClose={onCancelToolChange}
        title={t('controlLayers.vectorEdit.toolChangeTitle')}
        acceptCallback={onApplyToolChange}
        acceptButtonText={t('common.apply')}
        cancelCallback={onDiscardToolChange}
        cancelButtonText={t('controlLayers.vectorEdit.discardChanges')}
        useInert={false}
      >
        <Text>{t('controlLayers.vectorEdit.toolChangePrompt')}</Text>
      </ConfirmationAlertDialog>
      <Flex bg="base.800" borderRadius="base" p={4} minW={420} flexDir="column" gap={4} shadow="dark-lg">
        <Flex alignItems="center" justifyContent="space-between" gap={4}>
          <Heading size="md" color="base.300" userSelect="none">
            {t('controlLayers.vectorEdit.title')}
          </Heading>
          <Flex alignItems="center" gap={2} color="base.400">
            <Text fontSize="xs">{t('controlLayers.vectorEdit.deleteSelectionHint')}</Text>
            <Kbd fontSize="xs">{deletePathKeys.join('+')}</Kbd>
          </Flex>
        </Flex>
        <FormControl isDisabled={!selectedPointType || !canMutateEditSession}>
          <FormLabel m={0}>{t('controlLayers.vectorEdit.pointType')}</FormLabel>
          <RadioGroup
            value={selectedPointType === 'mixed' ? '' : (selectedPointType ?? '')}
            onChange={onPointTypeChange}
            size="sm"
          >
            <Flex alignItems="center" gap={4} color="base.300" wrap="wrap">
              <Radio value="corner">
                <Text>{t('controlLayers.vectorEdit.corner')}</Text>
              </Radio>
              <Radio value="smooth">
                <Text>{t('controlLayers.vectorEdit.smooth')}</Text>
              </Radio>
              <Radio value="symmetric">
                <Text>{t('controlLayers.vectorEdit.symmetric')}</Text>
              </Radio>
            </Flex>
          </RadioGroup>
        </FormControl>
        <Flex w="full" alignItems="center" gap={4} wrap="wrap">
          <Button
            onClick={onSmoothPath}
            isDisabled={
              !canMutateEditSession || (smoothTarget === 'selected' ? !canSmoothSelectedPoints : !canSmoothActivePath)
            }
            variant="ghost"
            size="sm"
            minW="unset"
            px={0}
          >
            {t('controlLayers.vectorEdit.smoothPath')}
          </Button>
          <Radio
            isChecked={smoothTarget === 'selected'}
            isReadOnly
            onClick={onSmoothSelectedPointsModeToggle}
            size="sm"
          >
            <Text>{t('controlLayers.vectorEdit.smoothSelectedPoints')}</Text>
          </Radio>
          <Button onClick={onSplitPoint} isDisabled={!canSplitPoint} variant="ghost" size="sm">
            {t('controlLayers.vectorEdit.splitPoint')}
          </Button>
          <Button onClick={onJoinPoints} isDisabled={!canJoinPoints} variant="ghost" size="sm">
            {t('controlLayers.vectorEdit.joinPoints')}
          </Button>
        </Flex>
        <ButtonGroup isAttached={false} size="sm" w="full" justifyContent="flex-end">
          <Button onClick={onReset} isDisabled={!canMutateEditSession} variant="ghost">
            {t('common.reset')}
          </Button>
          <Button onClick={onApply} variant="ghost">
            {t('common.apply')}
          </Button>
          <Button onClick={canvasManager.tool.tools.path.cancel} variant="ghost">
            {t('common.cancel')}
          </Button>
        </ButtonGroup>
      </Flex>
    </Fragment>
  );
});

VectorLayerEditFooter.displayName = 'VectorLayerEditFooter';
