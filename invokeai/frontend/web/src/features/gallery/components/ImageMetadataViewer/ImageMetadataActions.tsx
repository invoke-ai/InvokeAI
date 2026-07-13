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
  isCollectionMetadataHandler,
  isUnrecallableMetadataHandler,
  useCollectionMetadataDatum,
  useSingleMetadataDatum,
  useUnrecallableMetadataDatum,
} from 'features/metadata/parsing';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowBendUpLeftBold } from 'react-icons/pi';

type Props = {
  metadata?: unknown;
};

type ImageMetadataActionHandler =
  | UnrecallableMetadataHandler<any>
  | SingleMetadataHandler<any>
  | CollectionMetadataHandler<any[]>;

export const IMAGE_METADATA_ACTION_HANDLERS: ImageMetadataActionHandler[] = [
  ImageMetadataHandlers.GenerationMode,
  ImageMetadataHandlers.PositivePrompt,
  ImageMetadataHandlers.NegativePrompt,
  ImageMetadataHandlers.MainModel,
  ImageMetadataHandlers.VAEModel,
  ImageMetadataHandlers.Width,
  ImageMetadataHandlers.Height,
  ImageMetadataHandlers.Seed,
  ImageMetadataHandlers.Steps,
  ImageMetadataHandlers.Scheduler,
  ImageMetadataHandlers.CLIPSkip,
  ImageMetadataHandlers.CFGScale,
  ImageMetadataHandlers.CFGRescaleMultiplier,
  ImageMetadataHandlers.Guidance,
  ImageMetadataHandlers.FluxDypePreset,
  ImageMetadataHandlers.FluxDypeScale,
  ImageMetadataHandlers.FluxDypeExponent,
  ImageMetadataHandlers.DenoisingStrength,
  ImageMetadataHandlers.SeamlessX,
  ImageMetadataHandlers.SeamlessY,
  ImageMetadataHandlers.RefinerModel,
  ImageMetadataHandlers.RefinerCFGScale,
  ImageMetadataHandlers.RefinerPositiveAestheticScore,
  ImageMetadataHandlers.RefinerNegativeAestheticScore,
  ImageMetadataHandlers.RefinerScheduler,
  ImageMetadataHandlers.RefinerDenoisingStart,
  ImageMetadataHandlers.RefinerSteps,
  ImageMetadataHandlers.QwenImageComponentSource,
  ImageMetadataHandlers.QwenImageQuantization,
  ImageMetadataHandlers.QwenImageShift,
  ImageMetadataHandlers.ZImageShift,
  ImageMetadataHandlers.CanvasLayers,
  ImageMetadataHandlers.RefImages,
  ImageMetadataHandlers.KleinVAEModel,
  ImageMetadataHandlers.KleinQwen3EncoderModel,
  ImageMetadataHandlers.Krea2VAEModel,
  ImageMetadataHandlers.Krea2Qwen3VlEncoderModel,
  ImageMetadataHandlers.Krea2SeedVarianceEnabled,
  ImageMetadataHandlers.Krea2SeedVarianceStrength,
  ImageMetadataHandlers.Krea2SeedVarianceRandomizePercent,
  ImageMetadataHandlers.Krea2RebalanceEnabled,
  ImageMetadataHandlers.Krea2RebalanceMultiplier,
  ImageMetadataHandlers.Krea2RebalanceWeights,
  ImageMetadataHandlers.LoRAs,
];

export const ImageMetadataActions = memo((props: Props) => {
  const { metadata } = props;

  if (!metadata || Object.keys(metadata).length === 0) {
    return null;
  }

  return (
    <Flex flexDir="column" ps={8}>
      {IMAGE_METADATA_ACTION_HANDLERS.map((handler) => {
        if (isUnrecallableMetadataHandler(handler)) {
          return <UnrecallableMetadataDatum key={handler.type} metadata={metadata} handler={handler} />;
        }
        if (isCollectionMetadataHandler(handler)) {
          return <CollectionMetadataDatum key={handler.type} metadata={metadata} handler={handler} />;
        }
        return <SingleMetadataDatum key={handler.type} metadata={metadata} handler={handler} />;
      })}
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
    const { t } = useTranslation();
    const store = useAppStore();

    const { LabelComponent, ValueComponent } = handler;

    const onClick = useCallback(() => {
      handler.recall(data.value, store);
    }, [data.value, handler, store]);

    return (
      <Flex gap={2}>
        <IconButton
          aria-label={t('metadata.recallParameters')}
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
    const { t } = useTranslation();
    const store = useAppStore();

    const { LabelComponent, ValueComponent } = handler;

    const onClick = useCallback(() => {
      handler.recallOne(value, store);
    }, [handler, store, value]);

    return (
      <Flex gap={2}>
        <IconButton
          aria-label={t('metadata.recallParameters')}
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
