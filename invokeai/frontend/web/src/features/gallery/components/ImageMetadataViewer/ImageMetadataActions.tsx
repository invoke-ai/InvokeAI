import { Box, Flex, IconButton } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import type {
  CollectionMetadataHandler,
  SingleMetadataHandler,
  UnrecallableMetadataHandler,
} from 'features/metadata/parsing';
import {
  MetadataHanders,
  useCollectionMetadataDatum,
  useSingleMetadataDatum,
  useUnrecallableMetadataDatum,
} from 'features/metadata/parsing';
import { memo } from 'react';
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
      <UnrecallableMetadataDatum metadata={metadata} handler={MetadataHanders.CreatedBy} />
      <UnrecallableMetadataDatum metadata={metadata} handler={MetadataHanders.GenerationMode} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.PositivePrompt} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.NegativePrompt} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.PositiveStylePrompt} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.NegativeStylePrompt} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.MainModel} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.VAEModel} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.Width} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.Height} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.Seed} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.Steps} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.Scheduler} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.CFGScale} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.CFGRescaleMultiplier} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.Guidance} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.DenoisingStrength} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.SeamlessX} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.SeamlessY} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.RefinerModel} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.RefinerCFGScale} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.RefinerPositiveAestheticScore} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.RefinerNegativeAestheticScore} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.RefinerScheduler} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.RefinerDenoisingStart} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHanders.RefinerSteps} />
      <CollectionMetadataDatum metadata={metadata} handler={MetadataHanders.LoRAs} />
    </Flex>
  );
};

export default memo(ImageMetadataActions);

const UnrecallableMetadataDatum = typedMemo(
  <T,>({ metadata, handler }: { metadata: unknown; handler: UnrecallableMetadataHandler<T> }) => {
    const { data } = useUnrecallableMetadataDatum(metadata, handler);

    if (!data.isParsed) {
      return null;
    }

    if (data.isSuccess) {
      const { LabelComponent, ValueComponent } = handler;

      return (
        <Box as="span" lineHeight={1}>
          <LabelComponent value={data.value} />
          <ValueComponent value={data.value} />
        </Box>
      );
    }
  }
);
UnrecallableMetadataDatum.displayName = 'UnrecallableMetadataDatum';

const SingleMetadataDatum = typedMemo(
  <T,>({ metadata, handler }: { metadata: unknown; handler: SingleMetadataHandler<T> }) => {
    const { data, recall } = useSingleMetadataDatum(metadata, handler);

    if (!data.isParsed) {
      return null;
    }

    if (data.isSuccess) {
      const { LabelComponent, ValueComponent } = handler;
      return (
        <Flex gap={2}>
          <IconButton
            aria-label="Recall Parameter"
            icon={<PiArrowBendUpLeftBold />}
            size="xs"
            variant="ghost"
            onClick={recall}
          />
          <Box as="span" lineHeight={1}>
            <LabelComponent value={data.value} />
            <ValueComponent value={data.value} />
          </Box>
        </Flex>
      );
    }
  }
);
SingleMetadataDatum.displayName = 'SingleMetadataDatum';

const CollectionMetadataDatum = typedMemo(
  <T extends any[]>({ metadata, handler }: { metadata: unknown; handler: CollectionMetadataHandler<T> }) => {
    const { data, recallAll, recallItem } = useCollectionMetadataDatum(metadata, handler);

    if (!data.isParsed) {
      return null;
    }

    if (data.isSuccess) {
      const { LabelComponent, ValueComponent } = handler;

      return (
        <>
          {data.value.map((value, i) => (
            <Flex gap={2} key={i}>
              <IconButton
                aria-label="Recall Parameter"
                icon={<PiArrowBendUpLeftBold />}
                size="xs"
                variant="ghost"
                onClick={() => recallItem(value)}
              />
              <Box as="span" lineHeight={1}>
                <LabelComponent values={data.value} i={i} />
                <ValueComponent value={value} />
              </Box>
            </Flex>
          ))}
        </>
      );
    }
  }
);
CollectionMetadataDatum.displayName = 'CollectionMetadataDatum';
