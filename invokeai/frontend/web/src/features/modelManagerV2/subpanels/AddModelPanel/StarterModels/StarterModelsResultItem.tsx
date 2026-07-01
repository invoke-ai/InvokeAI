import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Badge, Flex, Text } from '@invoke-ai/ui-library';
import { negate } from 'es-toolkit/compat';
import { flattenStarterModel, useBuildModelInstallArg } from 'features/modelManagerV2/hooks/useBuildModelsToInstall';
import { useInstallModel } from 'features/modelManagerV2/hooks/useInstallModel';
import { ModelResultItemActions } from 'features/modelManagerV2/subpanels/AddModelPanel/ModelResultItemActions';
import ModelBaseBadge from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelBaseBadge';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { StarterModel } from 'services/api/types';

const starterModelResultItemSx: SystemStyleObject = {
  alignItems: 'start',
  justifyContent: 'space-between',
  w: '100%',
  py: 2,
  px: 1,
  gap: 2,
  borderBottomWidth: '1px',
  borderColor: 'base.700',
};

type Props = {
  starterModel: StarterModel;
};

export const StarterModelsResultItem = memo(({ starterModel }: Props) => {
  const { t } = useTranslation();
  const { getIsInstalled, buildModelInstallArg } = useBuildModelInstallArg();

  // The model is installed if it and all its dependencies are installed
  const isInstalled = useMemo(
    () => flattenStarterModel(starterModel).every(getIsInstalled),
    [getIsInstalled, starterModel]
  );

  // Build the install arguments for all models that are not installed
  const modelsToInstall = useMemo(
    () => flattenStarterModel(starterModel).filter(negate(getIsInstalled)).map(buildModelInstallArg),
    [getIsInstalled, starterModel, buildModelInstallArg]
  );

  const [installModel] = useInstallModel();

  const onClick = useCallback(() => {
    modelsToInstall.forEach(installModel);
    toast({
      status: 'info',
      title: t('modelManager.installingModel'),
      description: t('modelManager.installingXModels', { count: modelsToInstall.length }),
    });
  }, [modelsToInstall, installModel, t]);

  return (
    <Flex sx={starterModelResultItemSx}>
      <Flex fontSize="sm" flexDir="column">
        <Text fontWeight="semibold">{starterModel.name}</Text>
        <Text variant="subtext">{starterModel.description}</Text>
        <Flex gap={1} py={1} alignItems="center">
          <Badge h="min-content">{starterModel.type.replaceAll('_', ' ')}</Badge>
          <ModelBaseBadge base={starterModel.base} />
        </Flex>
      </Flex>
      <ModelResultItemActions handleInstall={onClick} isInstalled={isInstalled} />
    </Flex>
  );
});

StarterModelsResultItem.displayName = 'StarterModelsResultItem';
