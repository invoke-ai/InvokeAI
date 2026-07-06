import {
  Button,
  ButtonGroup,
  Flex,
  FormControl,
  FormLabel,
  Heading,
  Radio,
  RadioGroup,
  Spacer,
  Text,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { BezierPointType } from 'features/controlLayers/util/bezierPath';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const BEZIER_POINT_TYPES = ['corner', 'smooth', 'symmetric'] as const satisfies readonly BezierPointType[];

const isBezierPointType = (value: string): value is BezierPointType => {
  return BEZIER_POINT_TYPES.includes(value as BezierPointType);
};

export const VectorLayerEditFooter = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
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

  const onPointTypeChange = useCallback(
    (value: string) => {
      if (!isBezierPointType(value)) {
        return;
      }
      canvasManager.tool.tools.path.setActivePointType(value);
    },
    [canvasManager.tool.tools.path]
  );

  if (!editSession) {
    return null;
  }

  return (
    <Flex bg="base.800" borderRadius="base" p={4} minW={420} flexDir="column" gap={4} shadow="dark-lg">
      <Heading size="md" color="base.300" userSelect="none">
        {t('controlLayers.vectorEdit.title')}
      </Heading>
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
      <ButtonGroup isAttached={false} size="sm" w="full" alignItems="center">
        <Spacer />
        <Button onClick={canvasManager.tool.tools.path.acceptEditSession} variant="ghost">
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
