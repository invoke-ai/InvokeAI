import { MetadataItemView } from 'features/metadata/components/MetadataItemView';
import type { ControlNetConfigV2Metadata, MetadataHandlers } from 'features/metadata/types';
import { handlers } from 'features/metadata/util/handlers';
import { useCallback, useEffect, useMemo, useState } from 'react';

type Props = {
  metadata: unknown;
};

export const MetadataControlNetsV2 = ({ metadata }: Props) => {
  const [controlNets, setControlNets] = useState<ControlNetConfigV2Metadata[]>([]);

  useEffect(() => {
    const parse = async () => {
      try {
        const parsed = await handlers.controlNetsV2.parse(metadata);
        setControlNets(parsed);
      } catch (e) {
        setControlNets([]);
      }
    };
    parse();
  }, [metadata]);

  const label = useMemo(() => handlers.controlNetsV2.getLabel(), []);

  return (
    <>
      {controlNets.map((controlNet) => (
        <MetadataViewControlNet
          key={controlNet.id}
          label={label}
          controlNet={controlNet}
          handlers={handlers.controlNetsV2}
        />
      ))}
    </>
  );
};

const MetadataViewControlNet = ({
  label,
  controlNet,
  handlers,
}: {
  label: string;
  controlNet: ControlNetConfigV2Metadata;
  handlers: MetadataHandlers<ControlNetConfigV2Metadata[], ControlNetConfigV2Metadata>;
}) => {
  const onRecall = useCallback(() => {
    if (!handlers.recallItem) {
      return;
    }
    handlers.recallItem(controlNet, true);
  }, [handlers, controlNet]);

  const [renderedValue, setRenderedValue] = useState<React.ReactNode>(null);
  useEffect(() => {
    const _renderValue = async () => {
      if (!handlers.renderItemValue) {
        setRenderedValue(null);
        return;
      }
      const rendered = await handlers.renderItemValue(controlNet);
      setRenderedValue(rendered);
    };

    _renderValue();
  }, [handlers, controlNet]);

  return <MetadataItemView label={label} isDisabled={false} onRecall={onRecall} renderedValue={renderedValue} />;
};
