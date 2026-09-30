/* eslint-disable @typescript-eslint/no-explicit-any */
import { Box, Flex, IconButton } from '@invoke-ai/ui-library';
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
  ImageMetadataHandlers.T5EncoderModel,
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
  ImageMetadataHandlers.Ideogram4SamplerPreset,
  ImageMetadataHandlers.Ideogram4Steps,
  ImageMetadataHandlers.Ideogram4GuidanceScale,
  ImageMetadataHandlers.Ideogram4Mu,
  ImageMetadataHandlers.Ideogram4ColorPalette,
  ImageMetadataHandlers.Ideogram4Caption,
  ImageMetadataHandlers.CanvasLayers,
  ImageMetadataHandlers.RefImages,
  ImageMetadataHandlers.Flux1VAEModel,
  ImageMetadataHandlers.Flux2VAEModel,
  ImageMetadataHandlers.KleinQwen3EncoderModel,
  ImageMetadataHandlers.Flux2DevMistralEncoderModel,
  ImageMetadataHandlers.ZImageVAEModel,
  ImageMetadataHandlers.ZImageQwen3EncoderModel,
  ImageMetadataHandlers.ZImageQwen3SourceModel,
  ImageMetadataHandlers.AnimaVAEModel,
  ImageMetadataHandlers.AnimaQwen3EncoderModel,
  ImageMetadataHandlers.MiniMaxH3DurationSeconds,
  ImageMetadataHandlers.MiniMaxH3OutputMode,
  ImageMetadataHandlers.MiniMaxH3TransformerModel,
  ImageMetadataHandlers.MiniMaxH3TextEncoderModel,
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
    // `recall` comes from the hook rather than being `handler.recall`: the hook re-runs the handler's gate
    // before dispatching, so a row left over from a base switch cannot write into an inactive slot.
    const { data, recall } = useSingleMetadataDatum(metadata, handler);

    if (!data.isParsed) {
      return null;
    }

    if (data.isSuccess) {
      return <SingleMetadataParsed data={data} handler={handler} recall={recall} />;
    }
  }
);
SingleMetadataDatum.displayName = 'SingleMetadataDatum';

const SingleMetadataParsed = typedMemo(
  <T,>({
    data,
    handler,
    recall,
  }: {
    data: ParsedSuccessData<T>;
    handler: SingleMetadataHandler<T>;
    recall: (value: T) => void;
  }) => {
    const { t } = useTranslation();

    const { LabelComponent, ValueComponent } = handler;

    const onClick = useCallback(() => {
      recall(data.value);
    }, [data.value, recall]);

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
    // See `SingleMetadataDatum`: the hook's `recallOne` re-validates, `handler.recallOne` does not.
    const { data, recallOne } = useCollectionMetadataDatum(metadata, handler);

    if (!data.isParsed) {
      return null;
    }

    if (data.isSuccess) {
      return (
        <>
          {data.value.map((value, i) => (
            <CollectionMetadataParsed key={i} value={value} handler={handler} recallOne={recallOne} />
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
    handler,
    recallOne,
  }: {
    value: T[number];
    handler: CollectionMetadataHandler<T>;
    recallOne: (value: T[number]) => void;
  }) => {
    const { t } = useTranslation();

    const { LabelComponent, ValueComponent } = handler;

    const onClick = useCallback(() => {
      recallOne(value);
    }, [recallOne, value]);

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
