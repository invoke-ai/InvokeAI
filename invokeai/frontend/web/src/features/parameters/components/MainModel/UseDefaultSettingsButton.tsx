import { Flex, IconButton, ListItem, Text, UnorderedList } from '@invoke-ai/ui-library';
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
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSparkleFill } from 'react-icons/pi';
import { modelConfigsAdapterSelectors, useGetModelConfigsQuery } from 'services/api/endpoints/models';
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

  const defaultSettings = useMemo(() => {
    return modelConfig && isNonRefinerMainModelConfig(modelConfig) && modelConfig.default_settings
      ? modelConfig.default_settings
      : {};
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

      if (defaultVAE && vae && defaultVAE !== vae.key) {
        settings.push(t('modelManager.vae'));
      }

      if (defaultVAE && !vae && defaultVAE !== 'default') {
        settings.push(t('modelManager.vae'));
      }

      if (defaultVAEPrecision && defaultVAEPrecision !== vaePrecision) {
        settings.push(t('modelManager.vaePrecision'));
      }

      if (defaultCfg && defaultCfg !== cfg) {
        settings.push(t('parameters.cfgScale'));
      }

      if (defaultCfgRescale && defaultCfgRescale !== cfgRescale) {
        settings.push(t('parameters.cfgRescaleMultiplier'));
      }

      if (defaultSteps && defaultSteps !== steps) {
        settings.push(t('parameters.steps'));
      }

      if (defaultScheduler && defaultScheduler !== scheduler) {
        settings.push(t('parameters.scheduler'));
      }

      if (defaultWidth && defaultWidth !== width) {
        settings.push(t('parameters.width'));
      }

      if (defaultHeight && defaultHeight !== height) {
        settings.push(t('parameters.height'));
      }

      if (defaultGuidance && defaultGuidance !== guidance) {
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
    if (!hasDefaultSettings) {
      return t('modelManager.noDefaultSettings');
    }

    if (outOfSyncSettings.length === 0) {
      return t('modelManager.usingDefaultSettings');
    }

    return (
      <Flex direction="column" gap={3}>
        <Flex direction="column">
          <Text>{t('modelManager.defaultSettingsOutOfSync')}</Text>
          <UnorderedList>
            {outOfSyncSettings.map((setting) => (
              <ListItem key={setting}>{setting}</ListItem>
            ))}
          </UnorderedList>
        </Flex>
        <Text>{t('modelManager.restoreDefaultSettings')}</Text>
      </Flex>
    );
  }, [outOfSyncSettings, t, hasDefaultSettings]);

  const handleClickDefaultSettings = useCallback(() => {
    dispatch(setDefaultSettings());
  }, [dispatch]);

  return (
    <IconButton
      icon={<PiSparkleFill />}
      tooltip={tooltip}
      aria-label={t('modelManager.useDefaultSettings')}
      isDisabled={!model || !hasDefaultSettings}
      onClick={handleClickDefaultSettings}
      size="sm"
      variant="ghost"
      colorScheme={outOfSyncSettings.length ? 'warning' : 'invokeBlue'}
    />
  );
};
