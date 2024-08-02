import type { CanvasLayerState } from 'features/controlLayers/store/types';
import { MetadataItemView } from 'features/metadata/components/MetadataItemView';
import type { MetadataHandlers } from 'features/metadata/types';
import { handlers } from 'features/metadata/util/handlers';
import { useCallback, useEffect, useMemo, useState } from 'react';

type Props = {
  metadata: unknown;
};

export const MetadataLayers = ({ metadata }: Props) => {
  const [layers, setLayers] = useState<CanvasLayerState[]>([]);

  useEffect(() => {
    const parse = async () => {
      try {
        const parsed = await handlers.layers.parse(metadata);
        setLayers(parsed);
      } catch (e) {
        setLayers([]);
      }
    };
    parse();
  }, [metadata]);

  const label = useMemo(() => handlers.layers.getLabel(), []);

  return (
    <>
      {layers.map((layer) => (
        <MetadataViewLayer key={layer.id} label={label} layer={layer} handlers={handlers.layers} />
      ))}
    </>
  );
};

const MetadataViewLayer = ({
  label,
  layer,
  handlers,
}: {
  label: string;
  layer: CanvasLayerState;
  handlers: MetadataHandlers<CanvasLayerState[], CanvasLayerState>;
}) => {
  const onRecall = useCallback(() => {
    if (!handlers.recallItem) {
      return;
    }
    handlers.recallItem(layer, true);
  }, [handlers, layer]);

  const [renderedValue, setRenderedValue] = useState<React.ReactNode>(null);
  useEffect(() => {
    const _renderValue = async () => {
      if (!handlers.renderItemValue) {
        setRenderedValue(null);
        return;
      }
      const rendered = await handlers.renderItemValue(layer);
      setRenderedValue(rendered);
    };

    _renderValue();
  }, [handlers, layer]);

  return <MetadataItemView label={label} isDisabled={false} onRecall={onRecall} renderedValue={renderedValue} />;
};
