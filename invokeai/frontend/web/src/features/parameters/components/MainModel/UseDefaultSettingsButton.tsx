import { Flex, IconButton, ListItem, Text, UnorderedList } from '@invoke-ai/ui-library';
import { EMPTY_OBJECT } from 'app/store/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectCFGRescaleMultiplier,
  selectCFGScale,
  selectGuidance,
  selectModel,
  selectScheduler,
  selectSteps,
  selectVAE,
  selectVAEPrecision,
} from 'features/controlLayers/store/paramsSlice';
import { selectHeight, selectWidth } from 'features/controlLayers/store/selectors';
import { setDefaultSettings } from 'features/parameters/store/actions';
import { isNil } from 'lodash-es';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSparkleFill } from 'react-icons/pi';
import { modelConfigsAdapterSelectors, useGetModelConfigsQuery } from 'services/api/endpoints/models';
import type { S } from 'services/api/types';
import { isNonRefinerMainModelConfig } from 'services/api/types';

export const UseDefaultSettingsButton = () => {
  const model = useAppSelector(selectModel);
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { data: modelConfigs } = useGetModelConfigsQuery();

  const scheduler = useAppSelector(selectScheduler);
  const steps = useAppSelector(selectSteps);
  const vae = useAppSelector(selectVAE);
  const vaePrecision = useAppSelector(selectVAEPrecision);
  const width = useAppSelector(selectWidth);
  const height = useAppSelector(selectHeight);
  const guidance = useAppSelector(selectGuidance);
  const cfg = useAppSelector(selectCFGScale);
  const cfgRescale = useAppSelector(selectCFGRescaleMultiplier);

  const modelConfig = useMemo(() => {
    if (!modelConfigs) {
      return null;
    }
    if (model === null) {
      return null;
    }

    return modelConfigsAdapterSelectors.selectById(modelConfigs, model.key);
  }, [modelConfigs, model]);

  const hasDefaultSettings = useMemo(() => {
    const settings = modelConfig && isNonRefinerMainModelConfig(modelConfig) && modelConfig.default_settings;
    return settings && Object.values(settings).some((setting) => !!setting);
  }, [modelConfig]);

  const defaultSettings = useMemo<S['MainModelDefaultSettings']>(() => {
    return modelConfig && isNonRefinerMainModelConfig(modelConfig) && modelConfig.default_settings
      ? modelConfig.default_settings
      : EMPTY_OBJECT;
  }, [modelConfig]);

  const outOfSyncSettings = useMemo(() => {
    const settings = [];
    if (hasDefaultSettings) {
      const {
        vae: defaultVAE,
        vae_precision: defaultVAEPrecision,
        cfg_scale: defaultCfg,
        cfg_rescale_multiplier: defaultCfgRescale,
        steps: defaultSteps,
        scheduler: defaultScheduler,
        width: defaultWidth,
        height: defaultHeight,
        guidance: defaultGuidance,
      } = defaultSettings;

      if (!isNil(defaultVAE) && vae && defaultVAE !== vae.key) {
        settings.push(t('modelManager.vae'));
      }

      if (!isNil(defaultVAE) && !vae && defaultVAE !== 'default') {
        settings.push(t('modelManager.vae'));
      }

      if (!isNil(defaultVAEPrecision) && defaultVAEPrecision !== vaePrecision) {
        settings.push(t('modelManager.vaePrecision'));
      }

      if (!isNil(defaultCfg) && defaultCfg !== cfg) {
        settings.push(t('parameters.cfgScale'));
      }

      if (!isNil(defaultCfgRescale) && defaultCfgRescale !== cfgRescale) {
        settings.push(t('parameters.cfgRescaleMultiplier'));
      }

      if (!isNil(defaultSteps) && defaultSteps !== steps) {
        settings.push(t('parameters.steps'));
      }

      if (!isNil(defaultScheduler) && defaultScheduler !== scheduler) {
        settings.push(t('parameters.scheduler'));
      }

      if (!isNil(defaultWidth) && defaultWidth !== width) {
        settings.push(t('parameters.width'));
      }

      if (!isNil(defaultHeight) && defaultHeight !== height) {
        settings.push(t('parameters.height'));
      }

      if (!isNil(defaultGuidance) && defaultGuidance !== guidance) {
        settings.push(t('parameters.guidance'));
      }
    }
    return settings;
  }, [
    hasDefaultSettings,
    vae,
    vaePrecision,
    cfg,
    cfgRescale,
    steps,
    scheduler,
    width,
    height,
    guidance,
    t,
    defaultSettings,
  ]);

  const tooltip = useMemo(() => {
    if (!model) {
      return t('modelManager.noModelSelected');
    }

    if (!hasDefaultSettings) {
      return t('modelManager.noDefaultSettings');
    }

    if (outOfSyncSettings.length === 0) {
      return t('modelManager.usingDefaultSettings');
    }

    return (
      <Flex direction="column" gap={2}>
        <Text>{t('modelManager.defaultSettingsOutOfSync')}</Text>
        <UnorderedList>
          {outOfSyncSettings.map((setting) => (
            <ListItem key={setting}>{setting}</ListItem>
          ))}
        </UnorderedList>
        <Text>{t('modelManager.restoreDefaultSettings')}</Text>
      </Flex>
    );
  }, [model, hasDefaultSettings, outOfSyncSettings, t]);

  const handleClickDefaultSettings = useCallback(() => {
    dispatch(setDefaultSettings());
  }, [dispatch]);

  return (
    <IconButton
      icon={<PiSparkleFill />}
      tooltip={tooltip}
      aria-label={t('modelManager.useDefaultSettings')}
      isDisabled={!model || !hasDefaultSettings || outOfSyncSettings.length === 0}
      onClick={handleClickDefaultSettings}
      size="sm"
      variant="ghost"
      colorScheme="warning"
    />
  );
};
