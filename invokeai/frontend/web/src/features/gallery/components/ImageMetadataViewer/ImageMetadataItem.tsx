import { ExternalLink, Flex, IconButton, Text, Tooltip } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { get } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowBendUpLeftBold } from 'react-icons/pi';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';

type MetadataItemProps = {
  isLink?: boolean;
  label: string;
  metadata: unknown;
  propertyName: string;
  onRecall?: (value: unknown) => void;
  labelPosition?: string;
};

/**
 * Component to display an individual metadata item or parameter.
 */
const ImageMetadataItem = ({
  label,
  metadata,
  propertyName,
  onRecall: _onRecall,
  isLink,
  labelPosition,
}: MetadataItemProps) => {
  const { t } = useTranslation();
  const value = useMemo(() => get(metadata, propertyName), [metadata, propertyName]);
  const onRecall = useCallback(() => {
    if (!_onRecall) {
      return;
    }
    _onRecall(value);
  }, [_onRecall, value]);

  if (!value) {
    return null;
  }

  return (
    <Flex gap={2}>
      {_onRecall && (
        <Tooltip label={t('metadata.recallParameter', { parameter: label })}>
          <IconButton
            aria-label={t('metadata.recallParameter', { parameter: label })}
            icon={<PiArrowBendUpLeftBold />}
            size="xs"
            variant="ghost"
            fontSize={20}
            onClick={onRecall}
          />
        </Tooltip>
      )}
      <Flex direction={labelPosition ? 'column' : 'row'}>
        <Text fontWeight="semibold" whiteSpace="pre-wrap" pr={2}>
          {label}:
        </Text>
        {isLink ? (
          <ExternalLink href={value.toString()} label={value.toString()} />
        ) : (
          <Text overflowY="scroll" wordBreak="break-all">
            {value.toString()}
          </Text>
        )}
      </Flex>
    </Flex>
  );
};

export default memo(ImageMetadataItem);

type VAEMetadataItemProps = {
  label: string;
  modelKey?: string;
  onClick: () => void;
};

export const VAEMetadataItem = memo(({ label, modelKey, onClick }: VAEMetadataItemProps) => {
  const { data: modelConfig } = useGetModelConfigQuery(modelKey ?? skipToken);

  return (
    <ImageMetadataItem label={label} value={modelKey ? modelConfig?.name ?? modelKey : 'Default'} onClick={onClick} />
  );
});

VAEMetadataItem.displayName = 'VAEMetadataItem';

type ModelMetadataItemProps = {
  label: string;
  modelKey?: string;

  extra?: string;
  onClick: () => void;
};

export const ModelMetadataItem = memo(({ label, modelKey, extra, onClick }: ModelMetadataItemProps) => {
  const { data: modelConfig } = useGetModelConfigQuery(modelKey ?? skipToken);
  const value = useMemo(() => {
    if (modelConfig) {
      return `${modelConfig.name}${extra ?? ''}`;
    }
    return `${modelKey}${extra ?? ''}`;
  }, [extra, modelConfig, modelKey]);
  return <ImageMetadataItem label={label} value={value} onClick={onClick} />;
});

ModelMetadataItem.displayName = 'ModelMetadataItem';
