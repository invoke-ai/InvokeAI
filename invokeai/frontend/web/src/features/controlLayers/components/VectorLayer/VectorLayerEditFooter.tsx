import {
  Button,
  ButtonGroup,
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
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { BezierPointType } from 'features/controlLayers/util/bezierPath';
import { useHotkeyData } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo, useCallback, useState } from 'react';
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
  const activePointType = useAppSelector((state) => {
    if (!editSession || !editSession.activePathId || editSession.activePointIndex === null) {
      return null;
    }

    const canvas = selectCanvasSlice(state);
    const layer = canvas.vectorLayers.entities.find((entity) => entity.id === editSession.entityIdentifier.id);
    const path = layer?.paths.find((path) => path.id === editSession.activePathId);
    return path?.points[editSession.activePointIndex]?.type ?? null;
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
    if (!editSession?.activePathId || editSession.selectedPointIndices.length === 0) {
      return false;
    }

    const canvas = selectCanvasSlice(state);
    const layer = canvas.vectorLayers.entities.find((entity) => entity.id === editSession.entityIdentifier.id);
    const path = layer?.paths.find((path) => path.id === editSession.activePathId);
    return Boolean(
      path &&
      path.points.length >= 2 &&
      editSession.selectedPointIndices.some((pointIndex) => path.points[pointIndex] !== undefined)
    );
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

  if (!editSession) {
    return null;
  }

  const deletePathKeys = hotkeysData.canvas.hotkeys.deleteSelected?.platformKeys[0] ?? ['Delete'];

  return (
    <Flex bg="base.800" borderRadius="base" p={4} minW={420} flexDir="column" gap={4} shadow="dark-lg">
      <Flex alignItems="center" justifyContent="space-between" gap={4}>
        <Heading size="md" color="base.300" userSelect="none">
          {t('controlLayers.vectorEdit.title')}
        </Heading>
        <Flex alignItems="center" gap={2} color="base.400">
          <Text fontSize="xs">{t('controlLayers.vectorEdit.deleteActivePathHint')}</Text>
          <Kbd fontSize="xs">{deletePathKeys.join('+')}</Kbd>
        </Flex>
      </Flex>
      <FormControl isDisabled={!activePointType}>
        <FormLabel m={0}>{t('controlLayers.vectorEdit.pointType')}</FormLabel>
        <RadioGroup value={activePointType ?? undefined} onChange={onPointTypeChange} size="sm">
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
      <Flex w="full" alignItems="center" gap={4}>
        <Button
          onClick={onSmoothPath}
          isDisabled={smoothTarget === 'selected' ? !canSmoothSelectedPoints : !canSmoothActivePath}
          variant="ghost"
          size="sm"
          minW="unset"
          px={0}
        >
          {t('controlLayers.vectorEdit.smoothPath')}
        </Button>
        <Radio isChecked={smoothTarget === 'selected'} isReadOnly onClick={onSmoothSelectedPointsModeToggle} size="sm">
          <Text>{t('controlLayers.vectorEdit.smoothSelectedPoints')}</Text>
        </Radio>
      </Flex>
      <ButtonGroup isAttached={false} size="sm" w="full" justifyContent="flex-end">
        <Button onClick={onReset} variant="ghost">
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
  );
});

VectorLayerEditFooter.displayName = 'VectorLayerEditFooter';
