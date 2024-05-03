import { MetadataItemView } from 'features/metadata/components/MetadataItemView';
import type { MetadataHandlers, T2IAdapterConfigV2Metadata } from 'features/metadata/types';
import { handlers } from 'features/metadata/util/handlers';
import { useCallback, useEffect, useMemo, useState } from 'react';

type Props = {
  metadata: unknown;
};

export const MetadataT2IAdaptersV2 = ({ metadata }: Props) => {
  const [t2iAdapters, setT2IAdapters] = useState<T2IAdapterConfigV2Metadata[]>([]);

  useEffect(() => {
    const parse = async () => {
      try {
        const parsed = await handlers.t2iAdaptersV2.parse(metadata);
        setT2IAdapters(parsed);
      } catch (e) {
        setT2IAdapters([]);
      }
    };
    parse();
  }, [metadata]);

  const label = useMemo(() => handlers.t2iAdaptersV2.getLabel(), []);

  return (
    <>
      {t2iAdapters.map((t2iAdapter) => (
        <MetadataViewT2IAdapter
          key={t2iAdapter.id}
          label={label}
          t2iAdapter={t2iAdapter}
          handlers={handlers.t2iAdaptersV2}
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
  t2iAdapter: T2IAdapterConfigV2Metadata;
  handlers: MetadataHandlers<T2IAdapterConfigV2Metadata[], T2IAdapterConfigV2Metadata>;
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
