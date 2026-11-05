import type { CanvasLayerContract, TransformOptions } from '@workbench/canvas-engine/api';
import type { ToolFormProps } from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';

import { createListCollection, Flex, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { Select } from '@platform/ui/Select';
import { lookupDocumentLeaf } from '@workbench/canvas-engine/api';
import {
  useCanvasDocumentEditingLocked,
  useCanvasHasFloatingSelection,
  useTransformOptions,
  useTransformSession,
} from '@workbench/widgets/canvas/engineStoreHooks';
import { PropertyControlRow, PropertySwitchRow } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { fitLayerTransformToBbox } from '@workbench/widgets/layers/layerOps';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

const RESAMPLE_MODES = createListCollection({
  items: [
    { label: 'Bilinear', value: 'bilinear' },
    { label: 'Bicubic', value: 'bicubic' },
    { label: 'Hamming', value: 'hamming' },
    { label: 'Lanczos', value: 'lanczos' },
  ],
});

/** Transform-tool options act on its existing preview session, never on the layer-properties write path. */
export const TransformResamplingSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const session = useTransformSession(engine);
  const options = useTransformOptions(engine);
  const locked = useCanvasDocumentEditingLocked(engine);
  const hasFloat = useCanvasHasFloatingSelection(engine);
  const [error, setError] = useState<{ layerId: string; message: string } | null>(null);
  const selected = useActiveProjectSelector(
    (project): { layer: CanvasLayerContract; locked: boolean } | null => {
      const document = project.canvas.document;
      const leaf = document.selectedLayerId ? lookupDocumentLeaf(document, document.selectedLayerId) : null;
      return leaf ? { layer: leaf.layer, locked: leaf.effectiveLocked } : null;
    },
    (a, b) => a?.layer === b?.layer && a?.locked === b?.locked
  );
  const editable =
    !!session &&
    !!selected &&
    session.layerId === selected.layer.id &&
    !session.processing &&
    !selected.locked &&
    !locked &&
    !hasFloat;
  const pixelSource =
    selected &&
    (selected.layer.type === 'raster' || selected.layer.type === 'control') &&
    (selected.layer.source.type === 'image' || selected.layer.source.type === 'paint');
  const smoothingAvailable = editable && !!pixelSource;
  const fitCollection = useMemo(
    () =>
      createListCollection({
        items: [
          { label: t('widgets.transform.contain'), value: 'contain' },
          { label: t('widgets.transform.cover'), value: 'cover' },
          { label: t('widgets.transform.fill'), value: 'fill' },
        ],
      }),
    [t]
  );
  const resampleValue = useMemo(() => [options.resampleMode], [options.resampleMode]);
  const fitValue = useMemo(() => [options.fitMode], [options.fitMode]);
  const setSmoothing = useCallback(
    (smoothing: boolean) => engine.interaction.set('transformOptions', { ...options, smoothing }),
    [engine, options]
  );
  const setResampleMode = useCallback(
    ({ value }: { value: string[] }) => {
      if (value[0]) {
        engine.interaction.set('transformOptions', {
          ...options,
          resampleMode: value[0] as TransformOptions['resampleMode'],
        });
      }
    },
    [engine, options]
  );
  const setFitMode = useCallback(
    ({ value }: { value: string[] }) => {
      if (value[0]) {
        engine.interaction.set('transformOptions', { ...options, fitMode: value[0] as TransformOptions['fitMode'] });
      }
    },
    [engine, options]
  );
  const onFit = useCallback(() => {
    const document = engine.document.getDocument();
    if (!editable || !selected || !document || !session) {
      return;
    }
    const transform = fitLayerTransformToBbox(
      selected.layer,
      document.bbox,
      { ...document.bbox, width: document.width, height: document.height },
      options.fitMode
    );
    if (!transform) {
      setError({ layerId: selected.layer.id, message: t('widgets.canvas.structural.notReady') });
      return;
    }
    setError(null);
    engine.layers.updateTransformSession(transform);
  }, [editable, engine, options.fitMode, selected, session, t]);
  return (
    <>
      {error?.layerId === session?.layerId && error ? (
        <Text role="alert" color="fg.error">
          {error.message}
        </Text>
      ) : null}
      <PropertySwitchRow
        checked={options.smoothing}
        disabled={!smoothingAvailable}
        label={t('widgets.transform.smoothing')}
        onCheckedChange={setSmoothing}
      />
      <PropertyControlRow label={t('widgets.transform.resampleMode')} labelWidth="6rem">
        <Select
          aria-label={t('widgets.transform.resampleMode')}
          collection={RESAMPLE_MODES}
          disabled={!smoothingAvailable || !options.smoothing}
          gridColumn="2 / -1"
          value={resampleValue}
          onValueChange={setResampleMode}
        />
      </PropertyControlRow>
      <PropertyControlRow label={t('widgets.transform.fitMode')} labelWidth="6rem">
        <Flex gap="2" gridColumn="2 / -1" minW="0">
          <Select
            aria-label={t('widgets.transform.fitMode')}
            collection={fitCollection}
            disabled={!editable}
            flex="1"
            minW="0"
            value={fitValue}
            onValueChange={setFitMode}
          />
          <Button disabled={!editable} onClick={onFit}>
            {t('widgets.transform.fitToBbox')}
          </Button>
        </Flex>
      </PropertyControlRow>
      {options.smoothing ? (
        <Text color="fg.muted" fontSize="xs">
          {t('widgets.transform.smoothingHint')}
        </Text>
      ) : null}
    </>
  );
};
