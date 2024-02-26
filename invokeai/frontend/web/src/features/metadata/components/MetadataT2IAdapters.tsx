import { MetadataItemView } from 'features/metadata/components/MetadataItemView';
import type { MetadataHandlers, T2IAdapterConfigMetadata } from 'features/metadata/types';
import { handlers } from 'features/metadata/util/handlers';
import { useCallback, useEffect, useMemo, useState } from 'react';

type Props = {
  metadata: unknown;
};

export const MetadataT2IAdapters = ({ metadata }: Props) => {
  const [t2iAdapters, setT2IAdapters] = useState<T2IAdapterConfigMetadata[]>([]);

  useEffect(() => {
    const parse = async () => {
      try {
        const parsed = await handlers.t2iAdapters.parse(metadata);
        setT2IAdapters(parsed);
      } catch (e) {
        setT2IAdapters([]);
      }
    };
    parse();
  }, [metadata]);

  const label = useMemo(() => handlers.t2iAdapters.getLabel(), []);

  return (
    <>
      {t2iAdapters.map((t2iAdapter) => (
        <MetadataViewT2IAdapter
          key={t2iAdapter.id}
          label={label}
          t2iAdapter={t2iAdapter}
          handlers={handlers.t2iAdapters}
        />
      ))}
    </>
  );
};

const MetadataViewT2IAdapter = ({
  label,
  t2iAdapter,
  handlers,
}: {
  label: string;
  t2iAdapter: T2IAdapterConfigMetadata;
  handlers: MetadataHandlers<T2IAdapterConfigMetadata[], T2IAdapterConfigMetadata>;
}) => {
  const onRecall = useCallback(() => {
    if (!handlers.recallItem) {
      return;
    }
    handlers.recallItem(t2iAdapter, true);
  }, [handlers, t2iAdapter]);
  
  const [renderedValue, setRenderedValue] = useState<React.ReactNode>(null);
  useEffect(() => {
    const _renderValue = async () => {
      if (!handlers.renderItemValue) {
        setRenderedValue(null);
        return;
      }
      const rendered = await handlers.renderItemValue(t2iAdapter);
      setRenderedValue(rendered);
    };

    _renderValue();
  }, [handlers, t2iAdapter]);

  return <MetadataItemView label={label} isDisabled={false} onRecall={onRecall} renderedValue={renderedValue} />;
};
