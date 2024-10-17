import { Badge, Box, Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { flattenStarterModel, useBuildModelInstallArg } from 'features/modelManagerV2/hooks/useBuildModelsToInstall';
import { useInstallModel } from 'features/modelManagerV2/hooks/useInstallModel';
import ModelBaseBadge from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelBaseBadge';
import { toast } from 'features/toast/toast';
import { negate } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import type { StarterModel } from 'services/api/types';

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
    <Flex alignItems="center" justifyContent="space-between" w="100%" gap={3}>
      <Flex fontSize="sm" flexDir="column">
        <Flex gap={3}>
          <Badge h="min-content">{starterModel.type.replaceAll('_', ' ')}</Badge>
          <ModelBaseBadge base={starterModel.base} />
          <Text fontWeight="semibold">{starterModel.name}</Text>
        </Flex>
        <Text variant="subtext">{starterModel.description}</Text>
      </Flex>
      <Box>
        {isInstalled ? (
          <Badge>{t('common.installed')}</Badge>
        ) : (
          <IconButton aria-label={t('modelManager.install')} icon={<PiPlusBold />} onClick={onClick} size="sm" />
        )}
      </Box>
    </Flex>
  );
});

StarterModelsResultItem.displayName = 'StarterModelsResultItem';
