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
  MetadataHandlers,
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
      <UnrecallableMetadataDatum metadata={metadata} handler={MetadataHandlers.CreatedBy} />
      <UnrecallableMetadataDatum metadata={metadata} handler={MetadataHandlers.GenerationMode} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.PositivePrompt} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.NegativePrompt} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.PositiveStylePrompt} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.NegativeStylePrompt} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.MainModel} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.VAEModel} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.Width} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.Height} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.Seed} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.Steps} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.Scheduler} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.CFGScale} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.CFGRescaleMultiplier} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.Guidance} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.DenoisingStrength} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.SeamlessX} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.SeamlessY} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.RefinerModel} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.RefinerCFGScale} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.RefinerPositiveAestheticScore} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.RefinerNegativeAestheticScore} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.RefinerScheduler} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.RefinerDenoisingStart} />
      <SingleMetadataDatum metadata={metadata} handler={MetadataHandlers.RefinerSteps} />
      <CollectionMetadataDatum metadata={metadata} handler={MetadataHandlers.LoRAs} />
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
        <LabelComponent />
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
          <LabelComponent />
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
            <CollectionMetadataParsed key={i} value={value} handler={handler} />
          ))}
        </>
      );
    }
  }
);
CollectionMetadataDatum.displayName = 'CollectionMetadataDatum';

const CollectionMetadataParsed = typedMemo(
  <T extends any[]>({ value, handler }: { value: T[number]; handler: CollectionMetadataHandler<T> }) => {
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
          <LabelComponent />
          <ValueComponent value={value} />
        </Box>
      </Flex>
    );
  }
);
CollectionMetadataParsed.displayName = 'CollectionMetadataParsed';
