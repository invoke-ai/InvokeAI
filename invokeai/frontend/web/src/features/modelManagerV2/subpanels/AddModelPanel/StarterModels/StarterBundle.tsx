import { Button, Flex, Text, Tooltip } from '@invoke-ai/ui-library';
import { flattenStarterModel, useBuildModelInstallArg } from 'features/modelManagerV2/hooks/useBuildModelsToInstall';
import { isMainModelBase } from 'features/nodes/types/common';
import { MODEL_TYPE_SHORT_MAP } from 'features/parameters/types/constants';
import { toast } from 'features/toast/toast';
import { flatMap, negate, uniqWith } from 'lodash-es';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useInstallModelMutation } from 'services/api/endpoints/models';
import type { StarterModel } from 'services/api/types';

export const StarterBundle = ({ bundleName, bundle }: { bundleName: string; bundle: StarterModel[] }) => {
  const [installModel] = useInstallModelMutation();
  const { getIsInstalled, buildModelInstallArg } = useBuildModelInstallArg();
  const { t } = useTranslation();

  const modelsToInstall = useMemo(() => {
    // Flatten the models and remove duplicates, which is expected as models can have the same dependencies
    const flattenedModels = flatMap(bundle, flattenStarterModel);
    const uniqueModels = uniqWith(
      flattenedModels,
      (m1, m2) => m1.source === m2.source || (m1.name === m2.name && m1.base === m2.base && m1.type === m2.type)
    );
    // We want to install models that are not installed and skip models that are already installed
    const install = uniqueModels.filter(negate(getIsInstalled)).map(buildModelInstallArg);
    const skip = uniqueModels.filter(getIsInstalled).map(buildModelInstallArg);

    return { install, skip };
  }, [buildModelInstallArg, bundle, getIsInstalled]);

  const handleClickBundle = useCallback(() => {
    modelsToInstall.install.forEach(installModel);
    let description = t('modelManager.installingXModels', { count: modelsToInstall.install.length });
    if (modelsToInstall.skip.length > 1) {
      description += t('modelManager.skippingXDuplicates', { count: modelsToInstall.skip.length - 1 });
    }
    toast({
      status: 'info',
      title: t('modelManager.installingBundle'),
      description,
    });
  }, [modelsToInstall.install, modelsToInstall.skip.length, installModel, t]);

  return (
    <Tooltip
      label={
        <Flex flexDir="column">
          <Text>{t('modelManager.includesNModels', { n: bundle.length })}</Text>
        </Flex>
      }
    >
      <Button size="sm" onClick={handleClickBundle} py={6} isDisabled={modelsToInstall.install.length === 0}>
        <Flex flexDir="column">
          <Text>{isMainModelBase(bundleName) ? MODEL_TYPE_SHORT_MAP[bundleName] : bundleName}</Text>
          {modelsToInstall.install.length > 0 && (
            <Text fontSize="xs">
              ({bundle.length} {t('settings.models')})
            </Text>
          )}
          {modelsToInstall.install.length === 0 && <Text fontSize="xs">{t('common.installed')}</Text>}
        </Flex>
      </Button>
    </Tooltip>
  );
};
