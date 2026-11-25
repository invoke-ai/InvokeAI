import type { CanvasLayerContract } from '@workbench/canvas-engine/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { Stack, Text } from '@chakra-ui/react';
import { Button, ToggleIconButton } from '@platform/ui/Button';
import { ColorPicker } from '@platform/ui/ColorPicker';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { DEFAULT_VECTOR_GUIDE_COLOR, DEFAULT_VECTOR_STROKE_WIDTH } from '@workbench/canvas-engine/api';
import { PropertyControlRow, PropertySwitchRow } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { useColorSampler } from '@workbench/widgets/canvas/useColorSampler';
import {
  baselineConfig,
  useStructuralPreview,
  vectorTraceErrorKey,
} from '@workbench/widgets/canvas/useStructuralCommit';
import { PaintBucketIcon, SquareIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

type Engine = Pick<CanvasEngineHandle, 'document' | 'layers' | 'tools'>;
const TAPER_MARKS = [1, 100, 250, 500];
const formatPixels = (value: number) => `${value}px`;
const formatPercent = (value: number) => `${value}%`;

/** Layer appearance lives here; point commands belong to the Spline tool form. */
export const VectorLayerSettings = ({
  engine,
  layer,
}: {
  engine: Engine | null;
  layer: Extract<CanvasLayerContract, { type: 'vector' }>;
}) => {
  const { t } = useTranslation();
  const { preview, commit } = useStructuralPreview(engine);
  const sampleColor = useColorSampler(engine);
  const color = layer.guideColor ?? DEFAULT_VECTOR_GUIDE_COLOR;
  const onChange = useCallback(
    (guideColor: string) => {
      preview({ type: 'updateCanvasLayerConfig', id: layer.id, config: { layerType: 'vector', guideColor } });
    },
    [preview, layer.id]
  );
  const onChangeEnd = useCallback(
    (guideColor: string) => {
      const config = { layerType: 'vector', guideColor } as const;
      commit(t('widgets.layers.vector.guideColor'), (model, baseline) => {
        if (model.document.selectedLayerId !== layer.id) {
          return { status: 'unchanged' };
        }
        return model.prepare({ id: layer.id, type: 'patch-config', config, before: baselineConfig(baseline, config) });
      });
    },
    [commit, layer.id, t]
  );
  const resetGuideColor = useCallback(() => onChangeEnd(DEFAULT_VECTOR_GUIDE_COLOR), [onChangeEnd]);
  return (
    <Stack gap="2">
      <Text color="fg.muted" fontSize="sm">
        {t('widgets.layers.summary.vectorPaths', { count: layer.paths.length })}
      </Text>
      <PropertyControlRow label={t('widgets.layers.vector.guideColor')}>
        <ColorPicker
          aria-label={t('widgets.layers.vector.guideColor')}
          disabled={!engine || layer.isLocked || !layer.isEnabled}
          value={color}
          onSampleColor={sampleColor}
          onValueChange={onChange}
          onValueChangeEnd={onChangeEnd}
        />
        <Button
          size="xs"
          variant="outline"
          disabled={!engine || layer.isLocked || !layer.isEnabled || color === DEFAULT_VECTOR_GUIDE_COLOR}
          onClick={resetGuideColor}
        >
          {t('widgets.layers.vector.reset')}
        </Button>
      </PropertyControlRow>
      {engine ? <VectorTraceSettings key={layer.id} engine={engine} layer={layer} /> : null}
    </Stack>
  );
};

const VectorTraceSettings = ({
  engine,
  layer,
}: {
  engine: Engine;
  layer: Extract<CanvasLayerContract, { type: 'vector' }>;
}) => {
  const { t } = useTranslation();
  const sampleColor = useColorSampler(engine);
  const [failure, setFailure] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const { preview, commit } = useStructuralPreview(engine);
  const taperEnds = layer.traceTaperEnds ?? false;
  const taper = layer.traceTaper ?? 100;
  const label = (key: string) => t(`widgets.layers.vector.${key}`);
  const clampTaper = useCallback(
    (value: number) => Math.round(Math.min(500, Math.max(1, Number.isFinite(value) ? value : taper))),
    [taper]
  );
  const commitSettings = useCallback(
    (patch: {
      traceTaperEnds?: boolean;
      traceTaper?: number;
      fillColor?: string;
      strokeColor?: string;
      strokeWidth?: number;
      fillEnabled?: boolean;
      strokeEnabled?: boolean;
    }) => {
      const config = { layerType: 'vector', ...patch } as const;
      commit(t('widgets.layers.vector.traceTitle'), (model, baseline) => {
        if (model.document.selectedLayerId !== layer.id) {
          return { status: 'unchanged' };
        }
        return model.prepare({ id: layer.id, type: 'patch-config', config, before: baselineConfig(baseline, config) });
      });
    },
    [commit, layer.id, t]
  );
  const previewFillColor = useCallback(
    (fillColor: string) =>
      preview({ type: 'updateCanvasLayerConfig', id: layer.id, config: { layerType: 'vector', fillColor } }),
    [preview, layer.id]
  );
  const commitFillColor = useCallback((fillColor: string) => commitSettings({ fillColor }), [commitSettings]);
  const commitFillEnabled = useCallback((fillEnabled: boolean) => commitSettings({ fillEnabled }), [commitSettings]);
  const commitStrokeEnabled = useCallback(
    (strokeEnabled: boolean) => commitSettings({ strokeEnabled }),
    [commitSettings]
  );
  const previewStrokeColor = useCallback(
    (strokeColor: string) =>
      preview({ type: 'updateCanvasLayerConfig', id: layer.id, config: { layerType: 'vector', strokeColor } }),
    [preview, layer.id]
  );
  const commitStrokeColor = useCallback((strokeColor: string) => commitSettings({ strokeColor }), [commitSettings]);
  const clampStrokeWidth = useCallback(
    (value: number) => Math.round(Math.min(600, Math.max(1, Number.isFinite(value) ? value : layer.strokeWidth))),
    [layer.strokeWidth]
  );
  const previewStrokeWidth = useCallback(
    (value: number) =>
      preview({
        type: 'updateCanvasLayerConfig',
        id: layer.id,
        config: { layerType: 'vector', strokeWidth: clampStrokeWidth(value) },
      }),
    [preview, layer.id, clampStrokeWidth]
  );
  const commitStrokeWidth = useCallback(
    (value: number) => commitSettings({ strokeWidth: clampStrokeWidth(value) }),
    [commitSettings, clampStrokeWidth]
  );
  const commitTaperEnds = useCallback(
    (traceTaperEnds: boolean) => commitSettings({ traceTaperEnds }),
    [commitSettings]
  );
  const previewTaper = useCallback(
    (value: number) =>
      preview({
        id: layer.id,
        type: 'updateCanvasLayerConfig',
        config: { layerType: 'vector', traceTaper: clampTaper(value) },
      }),
    [preview, layer.id, clampTaper]
  );
  const commitTaper = useCallback(
    (value: number) => commitSettings({ traceTaper: clampTaper(value) }),
    [commitSettings, clampTaper]
  );
  const trace = useCallback(async () => {
    if (running) {
      engine.layers.cancelVectorTrace();
      return;
    }
    setFailure(null);
    setRunning(true);
    try {
      setFailure(vectorTraceErrorKey(await engine.layers.traceFillVectorLayer(layer.id)));
    } finally {
      setRunning(false);
    }
  }, [running, engine, layer.id]);
  const onTrace = useCallback(() => {
    void trace();
  }, [trace]);
  const disabled = !layer.isEnabled || layer.isLocked || layer.isHidden === true;
  return (
    <Stack borderTopWidth="1px" borderColor="border.subtle" gap="2" mt="2" pt="3">
      <Text color="fg.muted" fontWeight="600">
        {t('widgets.properties.groups.appearance')}
      </Text>
      <PropertyControlRow label={t('widgets.canvas.toolOptions.shapeFill')}>
        <ColorPicker
          aria-label={label('fillColor')}
          disabled={disabled || !layer.fillEnabled}
          value={layer.fillColor ?? '#ffffff'}
          onSampleColor={sampleColor}
          onValueChange={previewFillColor}
          onValueChangeEnd={commitFillColor}
        />
        <ToggleIconButton
          checked={layer.fillEnabled ?? false}
          disabled={disabled}
          icon={PaintBucketIcon}
          label={t('widgets.canvas.toolOptions.shapeFill')}
          onCheckedChange={commitFillEnabled}
        />
      </PropertyControlRow>
      <PropertyControlRow label={t('widgets.canvas.toolOptions.shapeStroke')}>
        <ColorPicker
          aria-label={label('traceColor')}
          disabled={disabled || layer.strokeEnabled === false}
          value={layer.strokeColor}
          onSampleColor={sampleColor}
          onValueChange={previewStrokeColor}
          onValueChangeEnd={commitStrokeColor}
        />
        <ToggleIconButton
          checked={layer.strokeEnabled ?? true}
          disabled={disabled}
          icon={SquareIcon}
          label={t('widgets.canvas.toolOptions.shapeStroke')}
          onCheckedChange={commitStrokeEnabled}
        />
      </PropertyControlRow>
      <ScrubberField
        label={label('strokeWidth')}
        disabled={disabled || layer.strokeEnabled === false}
        value={layer.strokeWidth}
        min={1}
        max={600}
        step={1}
        defaultValue={DEFAULT_VECTOR_STROKE_WIDTH}
        formatValue={formatPixels}
        onChange={previewStrokeWidth}
        onChangeEnd={commitStrokeWidth}
      />
      <PropertySwitchRow
        label={label('taperEnds')}
        disabled={disabled || layer.strokeEnabled === false}
        checked={taperEnds}
        onCheckedChange={commitTaperEnds}
      />
      <ScrubberField
        label={label('taper')}
        disabled={disabled || layer.strokeEnabled === false || !taperEnds}
        value={taper}
        min={1}
        max={500}
        step={10}
        defaultValue={100}
        marks={TAPER_MARKS}
        formatValue={formatPercent}
        onChange={previewTaper}
        onChangeEnd={commitTaper}
      />
      <Button
        size="sm"
        disabled={
          !running &&
          (disabled ||
            !layer.paths.some(
              (path) =>
                (layer.strokeEnabled !== false && path.points.length >= 2) ||
                (layer.fillEnabled && path.points.length >= 2)
            ))
        }
        onClick={onTrace}
      >
        {running ? t('common.cancel') : label('traceFill')}
      </Button>
      {failure ? (
        <Text color="fg.error" fontSize="sm">
          {label(failure)}
        </Text>
      ) : null}
    </Stack>
  );
};
