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
  ImageMetadataHandlers,
  useCollectionMetadataDatum,
  useSingleMetadataDatum,
  useUnrecallableMetadataDatum,
} from 'features/metadata/parsing';
import { memo, useCallback } from 'react';
import { PiArrowBendUpLeftBold } from 'react-icons/pi';

type Props = {
  metadata?: unknown;
};

export const ImageMetadataActions = memo((props: Props) => {
  const { metadata } = props;

  if (!metadata || Object.keys(metadata).length === 0) {
    return null;
  }

  return (
    <Flex flexDir="column" ps={8}>
      <UnrecallableMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.GenerationMode} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.PositivePrompt} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.NegativePrompt} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.MainModel} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.VAEModel} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.Width} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.Height} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.Seed} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.Steps} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.Scheduler} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.CLIPSkip} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.CFGScale} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.CFGRescaleMultiplier} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.Guidance} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.DenoisingStrength} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.SeamlessX} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.SeamlessY} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.RefinerModel} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.RefinerCFGScale} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.RefinerPositiveAestheticScore} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.RefinerNegativeAestheticScore} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.RefinerScheduler} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.RefinerDenoisingStart} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.RefinerSteps} />
      <SingleMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.CanvasLayers} />
      <CollectionMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.RefImages} />
      <CollectionMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.LoRAs} />
    </Flex>
  );
});

ImageMetadataActions.displayName = 'ImageMetadataActions';

export const UnrecallableMetadataDatum = typedMemo(
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
        <LabelComponent i18nKey={handler.i18nKey} />
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
          <LabelComponent i18nKey={handler.i18nKey} />
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
          <LabelComponent i18nKey={handler.i18nKey} />
          <ValueComponent value={value} />
        </Box>
      </Flex>
    );
  }
);
CollectionMetadataParsed.displayName = 'CollectionMetadataParsed';
