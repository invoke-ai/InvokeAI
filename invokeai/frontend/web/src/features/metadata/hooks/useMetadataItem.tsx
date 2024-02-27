import { Text } from '@invoke-ai/ui-library';
import type { MetadataHandlers } from 'features/metadata/types';
import { MetadataParseFailedToken, MetadataParsePendingToken } from 'features/metadata/util/parsers';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

const Pending = () => {
  const { t } = useTranslation();
  return <Text>{t('common.loading')}</Text>;
};

const Failed = () => {
  const { t } = useTranslation();
  return <Text>{t('metadata.parsingFailed')}</Text>;
};

export const useMetadataItem = <T,>(metadata: unknown, handlers: MetadataHandlers<T>) => {
  const [value, setValue] = useState<T | typeof MetadataParsePendingToken | typeof MetadataParseFailedToken>(
    MetadataParsePendingToken
  );
  const [renderedValue, setRenderedValue] = useState<React.ReactNode>(Pending);

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

  useEffect(() => {
    const _renderValue = async () => {
      if (value === MetadataParsePendingToken) {
        setRenderedValue(Pending);
        return;
      }
      if (value === MetadataParseFailedToken) {
        setRenderedValue(Failed);
        return;
      }

      const rendered = await handlers.renderValue(value);

      if (typeof rendered === 'string') {
        setRenderedValue(<Text>{rendered}</Text>);
        return;
      }
      setRenderedValue(rendered);
    };

    _renderValue();
  }, [handlers, value]);

  const onRecall = useCallback(() => {
    if (!handlers.recall || value === MetadataParsePendingToken || value === MetadataParseFailedToken) {
      return null;
    }
    handlers.recall(value, true);
  }, [handlers, value]);

  return { label, isDisabled, value, renderedValue, onRecall };
};
