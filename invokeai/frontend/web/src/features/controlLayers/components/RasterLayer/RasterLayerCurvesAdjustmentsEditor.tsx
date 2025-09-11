import { Box, Flex } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityAdapterContext } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { rasterLayerAdjustmentsCurvesUpdated } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice, selectEntity } from 'features/controlLayers/store/selectors';
import type { ChannelName, ChannelPoints } from 'features/controlLayers/store/types';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { RasterLayerCurvesAdjustmentsGraph } from './RasterLayerCurvesAdjustmentsGraph';

const DEFAULT_POINTS: ChannelPoints = [
  [0, 0],
  [255, 255],
];

type ChannelHistograms = Record<ChannelName, number[] | null>;

const calculateHistogramsFromImageData = (imageData: ImageData): ChannelHistograms | null => {
  try {
    const data = imageData.data;
    const len = data.length / 4;
    const master = new Array<number>(256).fill(0);
    const r = new Array<number>(256).fill(0);
    const g = new Array<number>(256).fill(0);
    const b = new Array<number>(256).fill(0);
    // sample every 4th pixel to lighten work
    for (let i = 0; i < len; i += 4) {
      const idx = i * 4;
      const rv = data[idx] as number;
      const gv = data[idx + 1] as number;
      const bv = data[idx + 2] as number;
      const m = Math.round(0.2126 * rv + 0.7152 * gv + 0.0722 * bv);
      if (m >= 0 && m < 256) {
        master[m] = (master[m] ?? 0) + 1;
      }
      if (rv >= 0 && rv < 256) {
        r[rv] = (r[rv] ?? 0) + 1;
      }
      if (gv >= 0 && gv < 256) {
        g[gv] = (g[gv] ?? 0) + 1;
      }
      if (bv >= 0 && bv < 256) {
        b[bv] = (b[bv] ?? 0) + 1;
      }
    }
    return {
      master,
      r,
      g,
      b,
    };
  } catch {
    return null;
  }
};

export const RasterLayerCurvesAdjustmentsEditor = memo(() => {
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext<'raster_layer'>();
  const adapter = useEntityAdapterContext<'raster_layer'>('raster_layer');
  const { t } = useTranslation();
  const selectLayer = useMemo(
    () => createSelector(selectCanvasSlice, (canvas) => selectEntity(canvas, entityIdentifier)),
    [entityIdentifier]
  );
  const layer = useAppSelector(selectLayer);
  const selectIsDisabled = useMemo(() => {
    return createSelector(
      selectCanvasSlice,
      (canvas) => selectEntity(canvas, entityIdentifier)?.adjustments?.enabled !== true
    );
  }, [entityIdentifier]);
  const isDisabled = useAppSelector(selectIsDisabled);

  const [histMaster, setHistMaster] = useState<number[] | null>(null);
  const [histR, setHistR] = useState<number[] | null>(null);
  const [histG, setHistG] = useState<number[] | null>(null);
  const [histB, setHistB] = useState<number[] | null>(null);

  const pointsMaster = layer?.adjustments?.curves.master ?? DEFAULT_POINTS;
  const pointsR = layer?.adjustments?.curves.r ?? DEFAULT_POINTS;
  const pointsG = layer?.adjustments?.curves.g ?? DEFAULT_POINTS;
  const pointsB = layer?.adjustments?.curves.b ?? DEFAULT_POINTS;

  const recalcHistogram = useCallback(() => {
    try {
      const rect = adapter.transformer.getRelativeRect();
      if (rect.width === 0 || rect.height === 0) {
        setHistMaster(Array(256).fill(0));
        setHistR(Array(256).fill(0));
        setHistG(Array(256).fill(0));
        setHistB(Array(256).fill(0));
        return;
      }
      const imageData = adapter.renderer.getImageData({ rect });
      const h = calculateHistogramsFromImageData(imageData);
      if (h) {
        setHistMaster(h.master);
        setHistR(h.r);
        setHistG(h.g);
        setHistB(h.b);
      }
    } catch {
      // ignore
    }
  }, [adapter]);

  useEffect(() => {
    recalcHistogram();
  }, [layer?.objects, layer?.adjustments, recalcHistogram]);

  const onChangePoints = useCallback(
    (channel: ChannelName, pts: ChannelPoints) => {
      dispatch(rasterLayerAdjustmentsCurvesUpdated({ entityIdentifier, channel, points: pts }));
    },
    [dispatch, entityIdentifier]
  );

  // Memoize per-channel change handlers to avoid inline lambdas in JSX
  const onChangeMaster = useCallback((pts: ChannelPoints) => onChangePoints('master', pts), [onChangePoints]);
  const onChangeR = useCallback((pts: ChannelPoints) => onChangePoints('r', pts), [onChangePoints]);
  const onChangeG = useCallback((pts: ChannelPoints) => onChangePoints('g', pts), [onChangePoints]);
  const onChangeB = useCallback((pts: ChannelPoints) => onChangePoints('b', pts), [onChangePoints]);

  return (
    <Flex
      direction="column"
      gap={2}
      px={3}
      pb={3}
      opacity={isDisabled ? 0.3 : 1}
      pointerEvents={isDisabled ? 'none' : 'auto'}
    >
      <Box display="grid" gridTemplateColumns="repeat(2, minmax(0, 1fr))" gap={4}>
        <RasterLayerCurvesAdjustmentsGraph
          title={t('controlLayers.adjustments.master')}
          channel="master"
          points={pointsMaster}
          histogram={histMaster}
          onChange={onChangeMaster}
        />
        <RasterLayerCurvesAdjustmentsGraph
          title={t('common.red')}
          channel="r"
          points={pointsR}
          histogram={histR}
          onChange={onChangeR}
        />
        <RasterLayerCurvesAdjustmentsGraph
          title={t('common.green')}
          channel="g"
          points={pointsG}
          histogram={histG}
          onChange={onChangeG}
        />
        <RasterLayerCurvesAdjustmentsGraph
          title={t('common.blue')}
          channel="b"
          points={pointsB}
          histogram={histB}
          onChange={onChangeB}
        />
      </Box>
    </Flex>
  );
});

RasterLayerCurvesAdjustmentsEditor.displayName = 'RasterLayerCurvesEditor';
