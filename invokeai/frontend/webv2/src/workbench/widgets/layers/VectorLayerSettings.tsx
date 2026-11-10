import type { CanvasLayerContract } from '@workbench/canvas-engine/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { Stack, Text } from '@chakra-ui/react';
import { PaintBucketIcon, SquareIcon } from 'lucide-react';
import { ToggleIconButton } from '@platform/ui/Button';
import { Button } from '@platform/ui/Button';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { ColorPicker } from '@platform/ui/ColorPicker';
import { DEFAULT_VECTOR_GUIDE_COLOR } from '@workbench/canvas-engine/api';
import { PropertyControlRow, PropertySwitchRow } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { useBrushOptions } from '@workbench/widgets/canvas/engineStoreHooks';
import { useActiveColorCommands } from '@workbench/widgets/canvas/color-system/useActiveColors';
import { useColorSampler } from '@workbench/widgets/canvas/useColorSampler';
import { baselineConfig, useStructuralPreview } from '@workbench/widgets/canvas/useStructuralCommit';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

type Engine = Pick<CanvasEngineHandle, 'document' | 'layers' | 'tools' | 'interaction'>;

/** Layer appearance lives here; point commands belong to the Spline tool form. */
export const VectorLayerSettings = ({ engine, layer }: {
  engine: Engine | null;
  layer: Extract<CanvasLayerContract, { type: 'vector' }>;
}) => {
  const { t } = useTranslation();
  const { preview, commit } = useStructuralPreview(engine);
  const sampleColor = useColorSampler(engine);
  const color = layer.guideColor ?? DEFAULT_VECTOR_GUIDE_COLOR;
  const onChange = useCallback((guideColor: string) => {
    preview({ type: 'updateCanvasLayerConfig', id: layer.id, config: { layerType: 'vector', guideColor } });
  }, [preview, layer.id]);
  const onChangeEnd = useCallback((guideColor: string) => {
    const config = { layerType: 'vector', guideColor } as const;
    commit(t('widgets.layers.vector.guideColor'), (model, baseline) => {
      if (model.document.selectedLayerId !== layer.id) { return { status: 'unchanged' }; }
      return model.prepare({ id: layer.id, type: 'patch-config', config, before: baselineConfig(baseline, config) });
    });
  }, [commit, layer.id, t]);
  return (
    <Stack gap="2">
      <Text color="fg.muted" fontSize="sm">{t('widgets.layers.summary.vectorPaths', { count: layer.paths.length })}</Text>
      <PropertyControlRow label={t('widgets.layers.vector.guideColor')}>
        <ColorPicker aria-label={t('widgets.layers.vector.guideColor')}
          disabled={!engine || layer.isLocked || !layer.isEnabled} value={color}
          onSampleColor={sampleColor} onValueChange={onChange} onValueChangeEnd={onChangeEnd} />
        <Button size="xs" variant="outline" disabled={!engine || layer.isLocked || !layer.isEnabled || color === DEFAULT_VECTOR_GUIDE_COLOR}
          onClick={() => onChangeEnd(DEFAULT_VECTOR_GUIDE_COLOR)}>{t('widgets.layers.vector.reset')}</Button>
      </PropertyControlRow>
      {engine ? <VectorTraceSettings engine={engine} layer={layer} /> : null}
    </Stack>
  );
};

const VectorTraceSettings = ({ engine, layer }: { engine: Engine; layer: Extract<CanvasLayerContract, { type: 'vector' }> }) => {
  const { t } = useTranslation();
  const brush = useBrushOptions(engine);
  const colorCommands = useActiveColorCommands();
  const sampleColor = useColorSampler(engine);
  const [failed, setFailed] = useState(false);
  const { preview, commit } = useStructuralPreview(engine);
  const taperEnds = layer.traceTaperEnds ?? false;
  const taper = layer.traceTaper ?? 100;
  const label = (key: string) => t(`widgets.layers.vector.${key}`);
  const clampTaper = (value: number) => Math.round(Math.min(500, Math.max(1, Number.isFinite(value) ? value : taper)));
  const commitSettings = (patch: { traceTaperEnds?: boolean; traceTaper?: number; fillColor?: string; fillEnabled?: boolean; strokeEnabled?: boolean }) => {
    const config = { layerType: 'vector', ...patch } as const;
    commit(label('traceTitle'), (model, baseline) => {
      if (model.document.selectedLayerId !== layer.id) { return { status: 'unchanged' }; }
      return model.prepare({ id: layer.id, type: 'patch-config', config, before: baselineConfig(baseline, config) });
    });
  };
  const trace = () => setFailed(!engine.layers.traceFillVectorLayer(layer.id));
  const disabled = !layer.isEnabled || layer.isLocked || layer.isHidden === true;
  return (
    <Stack borderTopWidth="1px" borderColor="border.subtle" gap="2" mt="2" pt="3">
      <Text color="fg.muted" fontWeight="600">{t('widgets.properties.groups.appearance')}</Text>
      <PropertyControlRow label={t('widgets.canvas.toolOptions.shapeFill')}>
        <ColorPicker aria-label={label('fillColor')} disabled={disabled || !layer.fillEnabled} value={layer.fillColor ?? '#ffffff'}
          onSampleColor={sampleColor}
          onValueChange={(fillColor) => preview({ type: 'updateCanvasLayerConfig', id: layer.id, config: { layerType: 'vector', fillColor } })}
          onValueChangeEnd={(fillColor) => commitSettings({ fillColor })} />
        <ToggleIconButton checked={layer.fillEnabled ?? false} disabled={disabled} icon={PaintBucketIcon}
          label={t('widgets.canvas.toolOptions.shapeFill')} onCheckedChange={(fillEnabled) => commitSettings({ fillEnabled })} />
      </PropertyControlRow>
      <PropertyControlRow label={t('widgets.canvas.toolOptions.shapeStroke')}>
        <ColorPicker aria-label={label('traceColor')} disabled={disabled || layer.strokeEnabled === false} value={brush.color}
          onSampleColor={sampleColor} onValueChange={colorCommands.applySampledColor} onValueChangeEnd={colorCommands.applySampledColor} />
        <ToggleIconButton checked={layer.strokeEnabled ?? true} disabled={disabled} icon={SquareIcon}
          label={t('widgets.canvas.toolOptions.shapeStroke')} onCheckedChange={(strokeEnabled) => commitSettings({ strokeEnabled })} />
      </PropertyControlRow>
      <ScrubberField label={label('strokeWidth')} disabled={disabled || layer.strokeEnabled === false} value={brush.size} min={1} max={600} step={1}
        defaultValue={50} formatValue={(value) => `${value}px`}
        onChange={(size) => engine.interaction.set('brushOptions', { ...engine.interaction.get('brushOptions'), size: Math.round(size) })} />
      <PropertySwitchRow label={label('taperEnds')} disabled={disabled || layer.strokeEnabled === false} checked={taperEnds}
        onCheckedChange={(traceTaperEnds) => commitSettings({ traceTaperEnds })} />
      <ScrubberField label={label('taper')} disabled={disabled || layer.strokeEnabled === false || !taperEnds} value={taper} min={1} max={500} step={10}
        defaultValue={100} marks={[1, 100, 250, 500]} formatValue={(value) => `${value}%`}
        onChange={(value) => preview({ id: layer.id, type: 'updateCanvasLayerConfig', config: { layerType: 'vector', traceTaper: clampTaper(value) } })}
        onChangeEnd={(value) => commitSettings({ traceTaper: clampTaper(value) })} />
      <Button size="sm" disabled={disabled || !layer.paths.some((path) => (layer.strokeEnabled !== false && path.points.length >= 2) || (layer.fillEnabled && path.points.length >= 2))} onClick={trace}>{label('traceFill')}</Button>
      {failed ? <Text color="fg.error" fontSize="sm">{label('traceFailed')}</Text> : null}
    </Stack>
  );
};
