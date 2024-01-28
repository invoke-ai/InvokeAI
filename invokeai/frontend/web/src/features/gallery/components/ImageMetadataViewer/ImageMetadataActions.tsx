import type {
  ControlNetMetadataItem,
  CoreMetadata,
  IPAdapterMetadataItem,
  LoRAMetadataItem,
  T2IAdapterMetadataItem,
} from 'features/nodes/types/metadata';
import { useRecallParameters } from 'features/parameters/hooks/useRecallParameters';
import {
  isParameterControlNetModel,
  isParameterLoRAModel,
  isParameterT2IAdapterModel,
} from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import ImageMetadataItem from './ImageMetadataItem';

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
    recallCfgRescaleMultiplier,
    recallModel,
    recallScheduler,
    recallVaeModel,
    recallSteps,
    recallWidth,
    recallHeight,
    recallStrength,
    recallHrfEnabled,
    recallHrfStrength,
    recallHrfMethod,
    recallLoRA,
    recallControlNet,
    recallIPAdapter,
    recallT2IAdapter,
    recallSDXLPositiveStylePrompt,
    recallSDXLNegativeStylePrompt,
  } = useRecallParameters();

  const handleRecallPositivePrompt = useCallback(() => {
    recallPositivePrompt(metadata?.positive_prompt);
  }, [metadata?.positive_prompt, recallPositivePrompt]);

  const handleRecallNegativePrompt = useCallback(() => {
    recallNegativePrompt(metadata?.negative_prompt);
  }, [metadata?.negative_prompt, recallNegativePrompt]);

  const handleRecallSDXLPositiveStylePrompt = useCallback(() => {
    recallSDXLPositiveStylePrompt(metadata?.positive_style_prompt);
  }, [metadata?.positive_style_prompt, recallSDXLPositiveStylePrompt]);

  const handleRecallSDXLNegativeStylePrompt = useCallback(() => {
    recallSDXLNegativeStylePrompt(metadata?.negative__style_prompt);
  }, [metadata?.negative__style_prompt, recallSDXLNegativeStylePrompt]);

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

  const handleRecallVaeModel = useCallback(() => {
    recallVaeModel(metadata?.vae);
  }, [metadata?.vae, recallVaeModel]);

  const handleRecallSteps = useCallback(() => {
    recallSteps(metadata?.steps);
  }, [metadata?.steps, recallSteps]);

  const handleRecallCfgScale = useCallback(() => {
    recallCfgScale(metadata?.cfg_scale);
  }, [metadata?.cfg_scale, recallCfgScale]);

  const handleRecallCfgRescaleMultiplier = useCallback(() => {
    recallCfgRescaleMultiplier(metadata?.cfg_rescale_multiplier);
  }, [metadata?.cfg_rescale_multiplier, recallCfgRescaleMultiplier]);

  const handleRecallStrength = useCallback(() => {
    recallStrength(metadata?.strength);
  }, [metadata?.strength, recallStrength]);

  const handleRecallHrfEnabled = useCallback(() => {
    recallHrfEnabled(metadata?.hrf_enabled);
  }, [metadata?.hrf_enabled, recallHrfEnabled]);

  const handleRecallHrfStrength = useCallback(() => {
    recallHrfStrength(metadata?.hrf_strength);
  }, [metadata?.hrf_strength, recallHrfStrength]);

  const handleRecallHrfMethod = useCallback(() => {
    recallHrfMethod(metadata?.hrf_method);
  }, [metadata?.hrf_method, recallHrfMethod]);

  const handleRecallLoRA = useCallback(
    (lora: LoRAMetadataItem) => {
      recallLoRA(lora);
    },
    [recallLoRA]
  );

  const handleRecallControlNet = useCallback(
    (controlnet: ControlNetMetadataItem) => {
      recallControlNet(controlnet);
    },
    [recallControlNet]
  );

  const handleRecallIPAdapter = useCallback(
    (ipAdapter: IPAdapterMetadataItem) => {
      recallIPAdapter(ipAdapter);
    },
    [recallIPAdapter]
  );

  const handleRecallT2IAdapter = useCallback(
    (ipAdapter: T2IAdapterMetadataItem) => {
      recallT2IAdapter(ipAdapter);
    },
    [recallT2IAdapter]
  );

  const validControlNets: ControlNetMetadataItem[] = useMemo(() => {
    return metadata?.controlnets
      ? metadata.controlnets.filter((controlnet) => isParameterControlNetModel(controlnet.control_model))
      : [];
  }, [metadata?.controlnets]);

  const validIPAdapters: IPAdapterMetadataItem[] = useMemo(() => {
    return metadata?.ipAdapters
      ? metadata.ipAdapters.filter((ipAdapter) => isParameterControlNetModel(ipAdapter.ip_adapter_model))
      : [];
  }, [metadata?.ipAdapters]);

  const validT2IAdapters: T2IAdapterMetadataItem[] = useMemo(() => {
    return metadata?.t2iAdapters
      ? metadata.t2iAdapters.filter((t2iAdapter) => isParameterT2IAdapterModel(t2iAdapter.t2i_adapter_model))
      : [];
  }, [metadata?.t2iAdapters]);

  if (!metadata || Object.keys(metadata).length === 0) {
    return null;
  }

  return (
    <>
      {metadata.created_by && <ImageMetadataItem label={t('metadata.createdBy')} value={metadata.created_by} />}
      {metadata.generation_mode && (
        <ImageMetadataItem label={t('metadata.generationMode')} value={metadata.generation_mode} />
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
          label={t('metadata.negativePrompt')}
          labelPosition="top"
          value={metadata.negative_prompt}
          onClick={handleRecallNegativePrompt}
        />
      )}
      {metadata.positive_style_prompt && (
        <ImageMetadataItem
          label={t('sdxl.posStylePrompt')}
          labelPosition="top"
          value={metadata.positive_style_prompt}
          onClick={handleRecallSDXLPositiveStylePrompt}
        />
      )}
      {metadata.negative_style_prompt && (
        <ImageMetadataItem
          label={t('sdxl.negStylePrompt')}
          labelPosition="top"
          value={metadata.negative_style_prompt}
          onClick={handleRecallSDXLNegativeStylePrompt}
        />
      )}
      {metadata.seed !== undefined && metadata.seed !== null && (
        <ImageMetadataItem label={t('metadata.seed')} value={metadata.seed} onClick={handleRecallSeed} />
      )}
      {metadata.model !== undefined && metadata.model !== null && metadata.model.model_name && (
        <ImageMetadataItem label={t('metadata.model')} value={metadata.model.model_name} onClick={handleRecallModel} />
      )}
      {metadata.width && (
        <ImageMetadataItem label={t('metadata.width')} value={metadata.width} onClick={handleRecallWidth} />
      )}
      {metadata.height && (
        <ImageMetadataItem label={t('metadata.height')} value={metadata.height} onClick={handleRecallHeight} />
      )}
      {metadata.scheduler && (
        <ImageMetadataItem label={t('metadata.scheduler')} value={metadata.scheduler} onClick={handleRecallScheduler} />
      )}
      <ImageMetadataItem
        label={t('metadata.vae')}
        value={metadata.vae?.model_name ?? 'Default'}
        onClick={handleRecallVaeModel}
      />
      {metadata.steps && (
        <ImageMetadataItem label={t('metadata.steps')} value={metadata.steps} onClick={handleRecallSteps} />
      )}
      {metadata.cfg_scale !== undefined && metadata.cfg_scale !== null && (
        <ImageMetadataItem label={t('metadata.cfgScale')} value={metadata.cfg_scale} onClick={handleRecallCfgScale} />
      )}
      {metadata.cfg_rescale_multiplier !== undefined && metadata.cfg_rescale_multiplier !== null && (
        <ImageMetadataItem
          label={t('metadata.cfgRescaleMultiplier')}
          value={metadata.cfg_rescale_multiplier}
          onClick={handleRecallCfgRescaleMultiplier}
        />
      )}
      {metadata.strength && (
        <ImageMetadataItem label={t('metadata.strength')} value={metadata.strength} onClick={handleRecallStrength} />
      )}
      {metadata.hrf_enabled && (
        <ImageMetadataItem
          label={t('hrf.metadata.enabled')}
          value={metadata.hrf_enabled}
          onClick={handleRecallHrfEnabled}
        />
      )}
      {metadata.hrf_enabled && metadata.hrf_strength && (
        <ImageMetadataItem
          label={t('hrf.metadata.strength')}
          value={metadata.hrf_strength}
          onClick={handleRecallHrfStrength}
        />
      )}
      {metadata.hrf_enabled && metadata.hrf_method && (
        <ImageMetadataItem
          label={t('hrf.metadata.method')}
          value={metadata.hrf_method}
          onClick={handleRecallHrfMethod}
        />
      )}
      {metadata.loras &&
        metadata.loras.map((lora, index) => {
          if (isParameterLoRAModel(lora.lora)) {
            return (
              <ImageMetadataItem
                key={index}
                label="LoRA"
                value={`${lora.lora.model_name} - ${lora.weight}`}
                onClick={handleRecallLoRA.bind(null, lora)}
              />
            );
          }
        })}
      {validControlNets.map((controlnet, index) => (
        <ImageMetadataItem
          key={index}
          label="ControlNet"
          value={`${controlnet.control_model?.model_name} - ${controlnet.control_weight}`}
          onClick={handleRecallControlNet.bind(null, controlnet)}
        />
      ))}
      {validIPAdapters.map((ipAdapter, index) => (
        <ImageMetadataItem
          key={index}
          label="IP Adapter"
          value={`${ipAdapter.ip_adapter_model?.model_name} - ${ipAdapter.weight}`}
          onClick={handleRecallIPAdapter.bind(null, ipAdapter)}
        />
      ))}
      {validT2IAdapters.map((t2iAdapter, index) => (
        <ImageMetadataItem
          key={index}
          label="T2I Adapter"
          value={`${t2iAdapter.t2i_adapter_model?.model_name} - ${t2iAdapter.weight}`}
          onClick={handleRecallT2IAdapter.bind(null, t2iAdapter)}
        />
      ))}
    </>
  );
};

export default memo(ImageMetadataActions);
