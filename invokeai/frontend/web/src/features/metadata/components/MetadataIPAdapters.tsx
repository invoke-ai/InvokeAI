import { Text } from '@invoke-ai/ui-library';
import type { IPAdapterConfig } from 'features/controlAdapters/store/types';
import { MetadataItemView } from 'features/metadata/components/MetadataItemView';
import type { MetadataHandlers } from 'features/metadata/types';
import { handlers } from 'features/metadata/util/handlers';
import { useCallback, useEffect, useMemo, useState } from 'react';

type Props = {
  metadata: unknown;
};

export const MetadataIPAdapters = ({ metadata }: Props) => {
  const [ipAdapters, setIPAdapters] = useState<IPAdapterConfig[]>([]);

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
        <MetadataViewIPAdapter
          key={ipAdapter.model.key}
          label={label}
          ipAdapter={ipAdapter}
          handlers={handlers.ipAdapters}
        />
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
  ipAdapter: IPAdapterConfig;
  handlers: MetadataHandlers<IPAdapterConfig[], IPAdapterConfig>;
}) => {
  const onRecall = useCallback(() => {
    if (!handlers.recallItem) {
      return;
    }
    handlers.recallItem(ipAdapter, true);
  }, [handlers, ipAdapter]);

  const renderedValue = useMemo(() => {
    if (!handlers.renderItemValue) {
      return null;
    }
    return <Text>{handlers.renderItemValue(ipAdapter)}</Text>;
  }, [handlers, ipAdapter]);

  return <MetadataItemView label={label} isDisabled={false} onRecall={onRecall} renderedValue={renderedValue} />;
};
