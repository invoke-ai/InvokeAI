import { MetadataItemView } from 'features/metadata/components/MetadataItemView';
import type { IPAdapterConfigMetadata, MetadataHandlers } from 'features/metadata/types';
import { handlers } from 'features/metadata/util/handlers';
import { useCallback, useEffect, useMemo, useState } from 'react';

type Props = {
  metadata: unknown;
};

export const MetadataIPAdapters = ({ metadata }: Props) => {
  const [ipAdapters, setIPAdapters] = useState<IPAdapterConfigMetadata[]>([]);

  useEffect(() => {
    const parse = async () => {
      try {
        const parsed = await handlers.ipAdapters.parse(metadata);
        setIPAdapters(parsed);
      } catch (e) {
        setIPAdapters([]);
      }
    };
    parse();
  }, [metadata]);

  const label = useMemo(() => handlers.ipAdapters.getLabel(), []);

  return (
    <>
      {ipAdapters.map((ipAdapter) => (
        <MetadataViewIPAdapter key={ipAdapter.id} label={label} ipAdapter={ipAdapter} handlers={handlers.ipAdapters} />
      ))}
    </>
  );
};

const MetadataViewIPAdapter = ({
  label,
  ipAdapter,
  handlers,
}: {
  label: string;
  ipAdapter: IPAdapterConfigMetadata;
  handlers: MetadataHandlers<IPAdapterConfigMetadata[], IPAdapterConfigMetadata>;
}) => {
  const onRecall = useCallback(() => {
    if (!handlers.recallItem) {
      return;
    }
    handlers.recallItem(ipAdapter, true);
  }, [handlers, ipAdapter]);

  const [renderedValue, setRenderedValue] = useState<React.ReactNode>(null);
  useEffect(() => {
    const _renderValue = async () => {
      if (!handlers.renderItemValue) {
        setRenderedValue(null);
        return;
      }
      const rendered = await handlers.renderItemValue(ipAdapter);
      setRenderedValue(rendered);
    };

    _renderValue();
  }, [handlers, ipAdapter]);

  return <MetadataItemView label={label} isDisabled={false} onRecall={onRecall} renderedValue={renderedValue} />;
};
