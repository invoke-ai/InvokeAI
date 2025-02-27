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
  const [renderedValueInternal, setRenderedValueInternal] = useState<React.ReactNode>(null);

  useEffect(() => {
    const _parse = () => {
      handlers
        .parse(metadata)
        .then((parsed) => {
          setValue(parsed);
        })
        .catch(() => {
          setValue(MetadataParseFailedToken);
        });
    };
    _parse();
  }, [handlers, metadata]);

  const isDisabled = useMemo(() => value === MetadataParsePendingToken || value === MetadataParseFailedToken, [value]);

  const label = useMemo(() => handlers.getLabel(), [handlers]);

  useEffect(() => {
    const _renderValue = () => {
      if (value === MetadataParsePendingToken) {
        setRenderedValueInternal(null);
        return;
      }
      if (value === MetadataParseFailedToken) {
        setRenderedValueInternal(null);
        return;
      }

      handlers
        .renderValue(value)
        .then((rendered) => {
          setRenderedValueInternal(rendered);
        })
        .catch(() => {
          // no-op
        });
    };

    _renderValue();
  }, [handlers, value]);

  const renderedValue = useMemo(() => {
    if (value === MetadataParsePendingToken) {
      return <Pending />;
    }
    if (value === MetadataParseFailedToken) {
      return <Failed />;
    }
    return <Text>{renderedValueInternal}</Text>;
  }, [renderedValueInternal, value]);

  const onRecall = useCallback(() => {
    if (!handlers.recall || value === MetadataParsePendingToken || value === MetadataParseFailedToken) {
      return null;
    }
    handlers.recall(value, true).catch(() => {
      // no-op, the toast will show the error
    });
  }, [handlers, value]);

  const valueOrNull = useMemo(() => {
    if (value === MetadataParsePendingToken || value === MetadataParseFailedToken) {
      return null;
    }
    return value;
  }, [value]);

  return { label, isDisabled, value, renderedValue, onRecall, valueOrNull };
};
