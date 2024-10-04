import { Badge, Box, Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { useInstallModel } from 'features/modelManagerV2/hooks/useInstallModel';
import ModelBaseBadge from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelBaseBadge';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import type { GetStarterModelsResponse } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

type Props = {
  result: GetStarterModelsResponse[number];
  modelList: AnyModelConfig[];
};
export const StarterModelsResultItem = memo(({ result, modelList }: Props) => {
  const { t } = useTranslation();
  const allSources = useMemo(() => {
    const _allSources = [
      {
        source: result.source,
        config: {
          name: result.name,
          description: result.description,
          type: result.type,
          base: result.base,
          format: result.format,
        },
      },
    ];
    if (result.dependencies) {
      for (const d of result.dependencies) {
        _allSources.push({
          source: d.source,
          config: { name: d.name, description: d.description, type: d.type, base: d.base, format: d.format },
        });
      }
    }
    return _allSources;
  }, [result]);
  const [installModel] = useInstallModel();

  const onPointerUp = useCallback(() => {
    for (const { config, source } of allSources) {
      if (modelList.some((mc) => config.base === mc.base && config.name === mc.name && config.type === mc.type)) {
        continue;
      }
      installModel({ config, source });
    }
  }, [modelList, allSources, installModel]);

  return (
    <Flex alignItems="center" justifyContent="space-between" w="100%" gap={3}>
      <Flex fontSize="sm" flexDir="column">
        <Flex gap={3}>
          <Badge h="min-content">{result.type.replaceAll('_', ' ')}</Badge>
          <ModelBaseBadge base={result.base} />
          <Text fontWeight="semibold">{result.name}</Text>
        </Flex>
        <Text variant="subtext">{result.description}</Text>
      </Flex>
      <Box>
        {result.is_installed ? (
          <Badge>{t('common.installed')}</Badge>
        ) : (
          <IconButton aria-label={t('modelManager.install')} icon={<PiPlusBold />} onPointerUp={onPointerUp} size="sm" />
        )}
      </Box>
    </Flex>
  );
});

StarterModelsResultItem.displayName = 'StarterModelsResultItem';
