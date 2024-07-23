import { Button, Flex, ListItem, Text, UnorderedList } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { $installModelsTab } from 'features/modelManagerV2/subpanels/InstallModels';
import { tileControlnetModelChanged } from 'features/parameters/store/upscaleSlice';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useCallback, useEffect, useMemo } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { useControlNetModels } from 'services/api/hooks/modelsByType';

export const MultidiffusionWarning = () => {
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

  const warnings = useMemo(() => {
    const _warnings: string[] = [];
    if (!model) {
      _warnings.push(t('upscaling.mainModelDesc'));
    }
    if (!tileControlnetModel) {
      _warnings.push(t('upscaling.tileControlNetModelDesc'));
    }
    if (!upscaleModel) {
      _warnings.push(t('upscaling.upscaleModelDesc'));
    }
    return _warnings;
  }, [model, upscaleModel, tileControlnetModel, t]);

  const handleGoToModelManager = useCallback(() => {
    dispatch(setActiveTab('models'));
    $installModelsTab.set(3);
  }, [dispatch]);

  if (!warnings.length || isLoading || !shouldShowButton) {
    return null;
  }

  return (
    <Flex bg="error.500" borderRadius="base" padding={4} direction="column" fontSize="sm" gap={2}>
      <Text>
        <Trans
          i18nKey="upscaling.missingModelsWarning"
          components={{
            LinkComponent: (
              <Button size="sm" flexGrow={0} variant="link" color="base.50" onClick={handleGoToModelManager} />
            ),
          }}
        />
      </Text>
      <UnorderedList>
        {warnings.map((warning) => (
          <ListItem key={warning}>{warning}</ListItem>
        ))}
      </UnorderedList>
    </Flex>
  );
};
