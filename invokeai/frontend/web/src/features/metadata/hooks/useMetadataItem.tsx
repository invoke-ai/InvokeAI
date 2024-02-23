import { Text } from '@invoke-ai/ui-library';
import type { MetadataHandlers } from 'features/metadata/types';
import { MetadataParseFailedToken, MetadataParsePendingToken } from 'features/metadata/util/parsers';
import { useCallback, useEffect, useMemo, useState } from 'react';

export const useMetadataItem = <T,>(metadata: unknown, handlers: MetadataHandlers<T>) => {
  const [value, setValue] = useState<T | typeof MetadataParsePendingToken | typeof MetadataParseFailedToken>(
    MetadataParsePendingToken
  );

  useEffect(() => {
    const _parse = async () => {
      try {
        const parsed = await handlers.parse(metadata);
        setValue(parsed);
      } catch (e) {
        setValue(MetadataParseFailedToken);
      }
    };
    _parse();
  }, [handlers, metadata]);

  const isDisabled = useMemo(() => value === MetadataParsePendingToken || value === MetadataParseFailedToken, [value]);

  const label = useMemo(() => handlers.getLabel(), [handlers]);

  const renderedValue = useMemo(() => {
    if (value === MetadataParsePendingToken) {
      return <Text>Loading</Text>;
    }
    if (value === MetadataParseFailedToken) {
      return <Text>Parsing Failed</Text>;
    }

    const rendered = handlers.renderValue(value);

    if (typeof rendered === 'string') {
      return <Text>{rendered}</Text>;
    }
    return rendered;
  }, [handlers, value]);

  const onRecall = useCallback(() => {
    if (!handlers.recall || value === MetadataParsePendingToken || value === MetadataParseFailedToken) {
      return null;
    }
    handlers.recall(value, true);
  }, [handlers, value]);

  return { label, isDisabled, value, renderedValue, onRecall };
};
