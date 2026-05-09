import { Button, Flex, Link, ListItem, Text, UnorderedList } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { selectModel } from 'features/controlLayers/store/paramsSlice';
import { setInstallModelsTabByName } from 'features/modelManagerV2/store/installModelsStore';
import {
  selectTileControlNetModel,
  selectUpscaleModel,
  tileControlnetModelChanged,
} from 'features/parameters/store/upscaleSlice';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { useCallback, useEffect, useMemo } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { useGetSetupStatusQuery } from 'services/api/endpoints/auth';
import { useControlNetModels } from 'services/api/hooks/modelsByType';

export const UpscaleWarning = () => {
  const { t } = useTranslation();
  const model = useAppSelector(selectModel);
  const upscaleModel = useAppSelector(selectUpscaleModel);
  const tileControlnetModel = useAppSelector(selectTileControlNetModel);
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useControlNetModels();
  const { data: setupStatus } = useGetSetupStatusQuery();
  const user = useAppSelector(selectCurrentUser);

  const isMultiuser = setupStatus?.multiuser_enabled ?? false;
  const isAdmin = !isMultiuser || (user?.is_admin ?? false);
  const adminEmail = setupStatus?.admin_email ?? null;

  useEffect(() => {
    const validModel = modelConfigs.find((cnetModel) => {
      return cnetModel.base === model?.base && cnetModel.name.toLowerCase().includes('tile');
    });
    if (tileControlnetModel?.key !== validModel?.key) {
      dispatch(tileControlnetModelChanged(validModel || null));
    }
  }, [dispatch, model?.base, modelConfigs, tileControlnetModel?.key]);

  const isBaseModelCompatible = useMemo(() => {
    return model && ['sd-1', 'sdxl'].includes(model.base);
  }, [model]);

  const warnings = useMemo(() => {
    const _warnings: string[] = [];
    if (!isBaseModelCompatible) {
      return _warnings;
    }
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
  }, [isBaseModelCompatible, model, tileControlnetModel, upscaleModel, t]);

  const handleGoToModelManager = useCallback(() => {
    navigationApi.switchToTab('models');
    setInstallModelsTabByName('launchpad');
  }, []);

  if ((isBaseModelCompatible && warnings.length === 0) || isLoading) {
    return null;
  }

  const AdminEmailLink = adminEmail ? (
    <Link href={`mailto:${adminEmail}`} color="base.50">
      {adminEmail}
    </Link>
  ) : (
    <Text as="span" color="base.50">
      your administrator
    </Text>
  );

  return (
    <Flex bg="error.500" borderRadius="base" padding={4} direction="column" fontSize="sm" gap={2}>
      {!isBaseModelCompatible && <Text>{t('upscaling.incompatibleBaseModelDesc')}</Text>}
      {warnings.length > 0 && (
        <Text>
          {isAdmin ? (
            <Trans
              i18nKey="upscaling.missingModelsWarning"
              components={{
                LinkComponent: (
                  <Button size="sm" flexGrow={0} variant="link" color="base.50" onClick={handleGoToModelManager} />
                ),
              }}
            />
          ) : (
            <Trans i18nKey="upscaling.missingModelsWarningNonAdmin" components={{ AdminEmailLink }} />
          )}
        </Text>
      )}
      {warnings.length > 0 && (
        <UnorderedList>
          {warnings.map((warning) => (
            <ListItem key={warning}>{warning}</ListItem>
          ))}
        </UnorderedList>
      )}
    </Flex>
  );
};
