import type { ControlNetConfig } from 'features/controlAdapters/store/types';
import { MetadataItemView } from 'features/metadata/components/MetadataItemView';
import type { MetadataHandlers } from 'features/metadata/types';
import { handlers } from 'features/metadata/util/handlers';
import { useCallback, useEffect, useMemo, useState } from 'react';

type Props = {
  metadata: unknown;
};

export const MetadataControlNets = ({ metadata }: Props) => {
  const [controlNets, setControlNets] = useState<ControlNetConfig[]>([]);

  useEffect(() => {
    const parse = async () => {
      try {
        const parsed = await handlers.controlNets.parse(metadata);
        setControlNets(parsed);
      } catch (e) {
        setControlNets([]);
      }
    };
    parse();
  }, [metadata]);

  const label = useMemo(() => handlers.controlNets.getLabel(), []);

  return (
    <>
      {controlNets.map((controlNet) => (
        <MetadataViewControlNet
          key={controlNet.id}
          label={label}
          controlNet={controlNet}
          handlers={handlers.controlNets}
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
  controlNet: ControlNetConfig;
  handlers: MetadataHandlers<ControlNetConfig[], ControlNetConfig>;
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
