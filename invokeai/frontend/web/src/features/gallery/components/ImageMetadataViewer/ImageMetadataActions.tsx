/* eslint-disable @typescript-eslint/no-explicit-any */
import { Box, Flex, IconButton } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/storeHooks';
import { typedMemo } from 'common/util/typedMemo';
import type {
  CollectionMetadataHandler,
  ParsedSuccessData,
  SingleMetadataHandler,
  UnrecallableMetadataHandler,
} from 'features/metadata/parsing';
import {
  MetadataHanders,
  useCollectionMetadataDatum,
  useSingleMetadataDatum,
  useUnrecallableMetadataDatum,
} from 'features/metadata/parsing';
import { memo, useCallback } from 'react';
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
      return <UnrecallableMetadataParsed data={data} handler={handler} />;
    }
  }
);
UnrecallableMetadataDatum.displayName = 'UnrecallableMetadataDatum';

const UnrecallableMetadataParsed = typedMemo(
  <T,>({ data, handler }: { data: ParsedSuccessData<T>; handler: UnrecallableMetadataHandler<T> }) => {
    const { LabelComponent, ValueComponent } = handler;

    return (
      <Box as="span" lineHeight={1}>
        <LabelComponent value={data.value} />
        <ValueComponent value={data.value} />
      </Box>
    );
  }
);
UnrecallableMetadataParsed.displayName = 'UnrecallableMetadataParsed';

const SingleMetadataDatum = typedMemo(
  <T,>({ metadata, handler }: { metadata: unknown; handler: SingleMetadataHandler<T> }) => {
    const { data } = useSingleMetadataDatum(metadata, handler);

    if (!data.isParsed) {
      return null;
    }

    if (data.isSuccess) {
      return <SingleMetadataParsed data={data} handler={handler} />;
    }
  }
);
SingleMetadataDatum.displayName = 'SingleMetadataDatum';

const SingleMetadataParsed = typedMemo(
  <T,>({ data, handler }: { data: ParsedSuccessData<T>; handler: SingleMetadataHandler<T> }) => {
    const store = useAppStore();

    const { LabelComponent, ValueComponent } = handler;

    const onClick = useCallback(() => {
      handler.recall(data.value, store);
    }, [data.value, handler, store]);

    return (
      <Flex gap={2}>
        <IconButton
          aria-label="Recall Parameter"
          icon={<PiArrowBendUpLeftBold />}
          size="xs"
          variant="ghost"
          onClick={onClick}
        />
        <Box as="span" lineHeight={1}>
          <LabelComponent value={data.value} />
          <ValueComponent value={data.value} />
        </Box>
      </Flex>
    );
  }
);
SingleMetadataParsed.displayName = 'SingleMetadataParsed';

const CollectionMetadataDatum = typedMemo(
  <T extends any[]>({ metadata, handler }: { metadata: unknown; handler: CollectionMetadataHandler<T> }) => {
    const { data } = useCollectionMetadataDatum(metadata, handler);

    if (!data.isParsed) {
      return null;
    }

    if (data.isSuccess) {
      return (
        <>
          {data.value.map((value, i) => (
            <CollectionMetadataParsed key={i} value={value} i={i} handler={handler} data={data} />
          ))}
        </>
      );
    }
  }
);
CollectionMetadataDatum.displayName = 'CollectionMetadataDatum';

const CollectionMetadataParsed = typedMemo(
  <T extends any[]>({
    value,
    i,
    data,
    handler,
  }: {
    value: T[number];
    i: number;
    data: ParsedSuccessData<T>;
    handler: CollectionMetadataHandler<T>;
  }) => {
    const store = useAppStore();

    const { LabelComponent, ValueComponent } = handler;

    const onClick = useCallback(() => {
      handler.recallOne(value, store);
    }, [handler, store, value]);

    return (
      <Flex gap={2}>
        <IconButton
          aria-label="Recall Parameter"
          icon={<PiArrowBendUpLeftBold />}
          size="xs"
          variant="ghost"
          onClick={onClick}
        />
        <Box as="span" lineHeight={1}>
          <LabelComponent values={data.value} i={i} />
          <ValueComponent value={value} />
        </Box>
      </Flex>
    );
  }
);
CollectionMetadataParsed.displayName = 'CollectionMetadataParsed';
