import { Badge, Box, Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { useBuildModelsToInstall } from 'features/modelManagerV2/hooks/useBuildModelsToInstall';
import { useInstallModel } from 'features/modelManagerV2/hooks/useInstallModel';
import ModelBaseBadge from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelBaseBadge';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import type { GetStarterModelsResponse } from 'services/api/endpoints/models';

type Props = {
  model: GetStarterModelsResponse['starter_models'][number];
};
export const StarterModelsResultItem = memo(({ model }: Props) => {
  const { t } = useTranslation();
  const buildModelToInstall = useBuildModelsToInstall();

  const allSources = useMemo(() => {
    const _allSources = [];

    const result = buildModelToInstall(model);
    if (result) {
      _allSources.push(result);
    }

    if (model.dependencies) {
      for (const d of model.dependencies) {
        const result = buildModelToInstall(d);
        if (result) {
          _allSources.push(result);
        }
      }
    }
    return _allSources;
  }, [model, buildModelToInstall]);

  const [installModel] = useInstallModel();

  const onClick = useCallback(() => {
    for (const model of allSources) {
      installModel(model);
    }
  }, [allSources, installModel]);

  return (
    <Flex alignItems="center" justifyContent="space-between" w="100%" gap={3}>
      <Flex fontSize="sm" flexDir="column">
        <Flex gap={3}>
          <Badge h="min-content">{model.type.replaceAll('_', ' ')}</Badge>
          <ModelBaseBadge base={model.base} />
          <Text fontWeight="semibold">{model.name}</Text>
        </Flex>
        <Text variant="subtext">{model.description}</Text>
      </Flex>
      <Box>
        {model.is_installed ? (
          <Badge>{t('common.installed')}</Badge>
        ) : (
          <IconButton aria-label={t('modelManager.install')} icon={<PiPlusBold />} onClick={onClick} size="sm" />
        )}
      </Box>
    </Flex>
  );
});

StarterModelsResultItem.displayName = 'StarterModelsResultItem';
