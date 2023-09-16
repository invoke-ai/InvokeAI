import { CoreMetadata } from 'features/nodes/types/types';
import { useRecallParameters } from 'features/parameters/hooks/useRecallParameters';
import { memo, useCallback } from 'react';
import ImageMetadataItem from './ImageMetadataItem';
import { useTranslation } from 'react-i18next';

type Props = {
  metadata?: CoreMetadata;
};

const ImageMetadataActions = (props: Props) => {
  const { metadata } = props;

  const { t } = useTranslation();

  const {
    recallPositivePrompt,
    recallNegativePrompt,
    recallSeed,
    recallCfgScale,
    recallModel,
    recallScheduler,
    recallSteps,
    recallWidth,
    recallHeight,
    recallStrength,
  } = useRecallParameters();

  const handleRecallPositivePrompt = useCallback(() => {
    recallPositivePrompt(metadata?.positive_prompt);
  }, [metadata?.positive_prompt, recallPositivePrompt]);

  const handleRecallNegativePrompt = useCallback(() => {
    recallNegativePrompt(metadata?.negative_prompt);
  }, [metadata?.negative_prompt, recallNegativePrompt]);

  const handleRecallSeed = useCallback(() => {
    recallSeed(metadata?.seed);
  }, [metadata?.seed, recallSeed]);

  const handleRecallModel = useCallback(() => {
    recallModel(metadata?.model);
  }, [metadata?.model, recallModel]);

  const handleRecallWidth = useCallback(() => {
    recallWidth(metadata?.width);
  }, [metadata?.width, recallWidth]);

  const handleRecallHeight = useCallback(() => {
    recallHeight(metadata?.height);
  }, [metadata?.height, recallHeight]);

  const handleRecallScheduler = useCallback(() => {
    recallScheduler(metadata?.scheduler);
  }, [metadata?.scheduler, recallScheduler]);

  const handleRecallSteps = useCallback(() => {
    recallSteps(metadata?.steps);
  }, [metadata?.steps, recallSteps]);

  const handleRecallCfgScale = useCallback(() => {
    recallCfgScale(metadata?.cfg_scale);
  }, [metadata?.cfg_scale, recallCfgScale]);

  const handleRecallStrength = useCallback(() => {
    recallStrength(metadata?.strength);
  }, [metadata?.strength, recallStrength]);

  if (!metadata || Object.keys(metadata).length === 0) {
    return null;
  }

  return (
    <>
      {metadata.created_by && (
        <ImageMetadataItem
          label={t('metadata.createdBy')}
          value={metadata.created_by}
        />
      )}
      {metadata.generation_mode && (
        <ImageMetadataItem
          label={t('metadata.generationMode')}
          value={metadata.generation_mode}
        />
      )}
      {metadata.positive_prompt && (
        <ImageMetadataItem
          label={t('metadata.positivePrompt')}
          labelPosition="top"
          value={metadata.positive_prompt}
          onClick={handleRecallPositivePrompt}
        />
      )}
      {metadata.negative_prompt && (
        <ImageMetadataItem
          label={t('metadata.NegativePrompt')}
          labelPosition="top"
          value={metadata.negative_prompt}
          onClick={handleRecallNegativePrompt}
        />
      )}
      {metadata.seed !== undefined && metadata.seed !== null && (
        <ImageMetadataItem
          label={t('metadata.seed')}
          value={metadata.seed}
          onClick={handleRecallSeed}
        />
      )}
      {metadata.model !== undefined &&
        metadata.model !== null &&
        metadata.model.model_name && (
          <ImageMetadataItem
            label={t('metadata.model')}
            value={metadata.model.model_name}
            onClick={handleRecallModel}
          />
        )}
      {metadata.width && (
        <ImageMetadataItem
          label={t('metadata.width')}
          value={metadata.width}
          onClick={handleRecallWidth}
        />
      )}
      {metadata.height && (
        <ImageMetadataItem
          label={t('metadata.height')}
          value={metadata.height}
          onClick={handleRecallHeight}
        />
      )}
      {/* {metadata.threshold !== undefined && (
          <MetadataItem
            label={t('metadata.threshold')}
            value={metadata.threshold}
            onClick={() => dispatch(setThreshold(Number(metadata.threshold)))}
          />
        )}
        {metadata.perlin !== undefined && (
          <MetadataItem
            label={t('metadata.perlin')}
            value={metadata.perlin}
            onClick={() => dispatch(setPerlin(Number(metadata.perlin)))}
          />
        )} */}
      {metadata.scheduler && (
        <ImageMetadataItem
          label={t('metadata.scheduler')}
          value={metadata.scheduler}
          onClick={handleRecallScheduler}
        />
      )}
      {metadata.steps && (
        <ImageMetadataItem
          label={t('metadata.steps')}
          value={metadata.steps}
          onClick={handleRecallSteps}
        />
      )}
      {metadata.cfg_scale !== undefined && metadata.cfg_scale !== null && (
        <ImageMetadataItem
          label={t('metadata.cfgScale')}
          value={metadata.cfg_scale}
          onClick={handleRecallCfgScale}
        />
      )}
      {/* {metadata.variations && metadata.variations.length > 0 && (
          <MetadataItem
            label="{t('metadata.variations')}
            value={seedWeightsToString(metadata.variations)}
            onClick={() =>
              dispatch(
                setSeedWeights(seedWeightsToString(metadata.variations))
              )
            }
          />
        )}
        {metadata.seamless && (
          <MetadataItem
            label={t('metadata.seamless')}
            value={metadata.seamless}
            onClick={() => dispatch(setSeamless(metadata.seamless))}
          />
        )}
        {metadata.hires_fix && (
          <MetadataItem
            label={t('metadata.hiresFix')}
            value={metadata.hires_fix}
            onClick={() => dispatch(setHiresFix(metadata.hires_fix))}
          />
        )} */}

      {/* {init_image_path && (
          <MetadataItem
            label={t('metadata.initImage')}
            value={init_image_path}
            isLink
            onClick={() => dispatch(setInitialImage(init_image_path))}
          />
        )} */}
      {metadata.strength && (
        <ImageMetadataItem
          label={t('metadata.strength')}
          value={metadata.strength}
          onClick={handleRecallStrength}
        />
      )}
      {/* {metadata.fit && (
          <MetadataItem
            label={t('metadata.fit')}
            value={metadata.fit}
            onClick={() => dispatch(setShouldFitToWidthHeight(metadata.fit))}
          />
        )} */}
    </>
  );
};

export default memo(ImageMetadataActions);
