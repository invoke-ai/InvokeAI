import { Flex, Link, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { tileControlnetModelChanged } from 'features/parameters/store/upscaleSlice';
import { MODEL_TYPE_SHORT_MAP } from 'features/parameters/types/constants';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useControlNetModels } from 'services/api/hooks/modelsByType';

interface Props {
  usesTile: boolean;
}

export const UpscaleWarning = ({ usesTile }: Props) => {
  const { t } = useTranslation();
  const model = useAppSelector((s) => s.generation.model);
  const { tileControlnetModel, upscaleModel } = useAppSelector((s) => s.upscale);
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useControlNetModels();
  const disabledTabs = useAppSelector((s) => s.config.disabledTabs);
  const shouldShowButton = useMemo(() => !disabledTabs.includes('models'), [disabledTabs]);

  useEffect(() => {
    const validModel = modelConfigs.find((cnetModel) => {
      return cnetModel.base === model?.base && cnetModel.name.toLowerCase().includes('tile');
    });
    dispatch(tileControlnetModelChanged(validModel || null));
  }, [model?.base, modelConfigs, dispatch]);

  const warningText = useMemo(() => {
    if (!model) {
      return t('upscaling.warningNoMainModel');
    }
    if (!upscaleModel && !tileControlnetModel && usesTile) {
      return t('upscaling.warningNoTileOrUpscaleModel', { base_model: MODEL_TYPE_SHORT_MAP[model.base] });
    }
    if (!upscaleModel) {
      return t('upscaling.warningNoUpscaleModel');
    }
    if (!tileControlnetModel && usesTile) {
      return t('upscaling.warningNoTile', { base_model: MODEL_TYPE_SHORT_MAP[model.base] });
    }
  }, [model, upscaleModel, tileControlnetModel, usesTile, t]);

  const handleGoToModelManager = useCallback(() => {
    dispatch(setActiveTab('models'));
  }, [dispatch]);

  if (!warningText || isLoading || !shouldShowButton) {
    return <></>;
  }

  return (
    <Flex bg="error.500" borderRadius="base" padding="2" direction="column">
      <Text fontSize="xs" textAlign="center" display="inline-block">
        {t('upscaling.visit')}{' '}
        <Link fontWeight="bold" onClick={handleGoToModelManager}>
          {t('modelManager.modelManager')}
        </Link>{' '}
        {t('upscaling.toInstall')} {warningText}.
      </Text>
    </Flex>
  );
};
