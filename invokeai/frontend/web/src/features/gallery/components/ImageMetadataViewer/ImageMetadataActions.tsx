import type { FlexProps } from '@invoke-ai/ui-library';
import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { isPrimitive } from 'es-toolkit';
import { MetadataLoRAs } from 'features/metadata/components/MetadataLoRAs';
import { MetadataHanders, type MetadataHandler, useMetadata } from 'features/metadata/parsing';
import type { ReactNode } from 'react';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowBendUpLeftBold } from 'react-icons/pi';

type Props = {
  metadata?: unknown;
};

const ImageMetadataActions = (props: Props) => {
  const { metadata } = props;

  if (!metadata || Object.keys(metadata).length === 0) {
    return null;
  }

  return (
    <Flex flexDir="column" ps={8}>
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.CreatedBy} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.GenerationMode} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.PositivePrompt} flexDir="column" />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.NegativePrompt} flexDir="column" />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.PositiveStylePrompt} flexDir="column" />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.NegativeStylePrompt} flexDir="column" />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.NegativePrompt} flexDir="column" />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.MainModel} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.VAEModel} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.Width} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.Height} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.Seed} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.Steps} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.Scheduler} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.CFGScale} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.CFGRescaleMultiplier} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.Guidance} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.DenoisingStrength} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.SeamlessX} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.SeamlessY} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.RefinerModel} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.RefinerCFGScale} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.RefinerPositiveAestheticScore} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.RefinerNegativeAestheticScore} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.RefinerScheduler} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.RefinerDenoisingStart} />
      <MetadataItem2 metadata={metadata} handler={MetadataHanders.RefinerSteps} />
      <MetadataLoRAs metadata={metadata} />
    </Flex>
  );
};

export default memo(ImageMetadataActions);

const MetadataItem2 = typedMemo(
  <T,>({ metadata, handler, ...rest }: { metadata: unknown; handler: MetadataHandler<T> } & FlexProps) => {
    const { t } = useTranslation();
    const { data, recall } = useMetadata(metadata, handler);

    if (!data.isParsed) {
      return null;
    }

    if (data.isSuccess) {
      const label = handler.renderLabel(data.value, t);
      const value = handler.renderValue(data.value, t);

      return (
        <Flex gap={2}>
          <IconButton
            aria-label="Recall Parameter"
            icon={<PiArrowBendUpLeftBold />}
            size="xs"
            variant="ghost"
            onClick={recall}
          />
          <Flex {...rest}>
            <MetadataLabel label={label} />
            <MetadataValue value={value} />
          </Flex>
        </Flex>
      );
    }
  }
);
MetadataItem2.displayName = 'MetadataItem2';

const MetadataLabel = ({ label }: { label: ReactNode }) => {
  if (isPrimitive(label)) {
    return (
      <Text fontWeight="semibold" whiteSpace="pre-wrap" me={2}>
        {label}
      </Text>
    );
  } else {
    return <>{label}</>;
  }
};

const MetadataValue = ({ value }: { value: ReactNode }) => {
  if (isPrimitive(value)) {
    return <Text>{value}</Text>;
  }
  return <>{value}</>;
};
