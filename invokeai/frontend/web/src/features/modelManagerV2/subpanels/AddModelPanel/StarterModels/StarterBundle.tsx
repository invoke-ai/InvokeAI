import { Button, Flex, Text, Tooltip } from '@invoke-ai/ui-library';
import { useBuildModelsToInstall } from 'features/modelManagerV2/hooks/useBuildModelsToInstall';
import { isMainModelBase } from 'features/nodes/types/common';
import { MODEL_TYPE_SHORT_MAP } from 'features/parameters/types/constants';
import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { type GetStarterModelsResponse, useInstallModelMutation } from 'services/api/endpoints/models';

export const StarterBundle = ({
  bundleName,
  bundle,
}: {
  bundleName: string;
  bundle: GetStarterModelsResponse['starter_bundles'][number];
}) => {
  const [installModel] = useInstallModelMutation();
  const buildModelToInstall = useBuildModelsToInstall();
  const { t } = useTranslation();

  const modelsToInstall = useMemo(() => {
    const _modelsToInstall = [];
    const _modelsToSkip = [];
    for (let index = 0; index < bundle.length; index++) {
      const starterModel = bundle[index];
      if (!starterModel) {
        continue;
      }

      const result = buildModelToInstall(starterModel);
      if (result) {
        _modelsToInstall.push(result);
      } else {
        _modelsToSkip.push(result);
      }

      if (starterModel.dependencies) {
        for (const d of starterModel.dependencies) {
          const result = buildModelToInstall(d);
          if (result) {
            _modelsToInstall.push(result);
          } else {
            _modelsToSkip.push(result);
          }
        }
      }
    }

    return { install: _modelsToInstall, skip: _modelsToSkip };
  }, [bundle, buildModelToInstall]);

  const handleClickBundle = useCallback(async () => {
    for (let index = 0; index < modelsToInstall.install.length; index++) {
      const model = modelsToInstall.install[index];
      if (model) {
        await installModel(model).unwrap();
      }
    }
    toast({
      status: 'info',
      title: 'Bundle Installing',
      description: `Installing ${modelsToInstall.install.length}, skipping ${modelsToInstall.skip.length} duplicates`,
    });
  }, [modelsToInstall, installModel]);

  return (
    <Tooltip
      label={
        <Flex flexDir="column">
          <Text>{t('modelManager.includesNModels', { n: bundle.length })}</Text>
        </Flex>
      }
    >
      <Button flexDir="column" size="sm" onClick={handleClickBundle}>
        {isMainModelBase(bundleName) && MODEL_TYPE_SHORT_MAP[bundleName]}
      </Button>
    </Tooltip>
  );
};
