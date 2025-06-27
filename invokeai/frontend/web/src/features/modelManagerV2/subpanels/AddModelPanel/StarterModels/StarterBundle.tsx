import { Button, Flex, ListItem, Text, Tooltip, UnorderedList } from '@invoke-ai/ui-library';
import { useStarterBundleInstall } from 'features/modelManagerV2/hooks/useStarterBundleInstall';
import { isMainModelBase } from 'features/nodes/types/common';
import { MODEL_TYPE_SHORT_MAP } from 'features/parameters/types/constants';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { S } from 'services/api/types';

export const StarterBundle = ({ bundle }: { bundle: S['StarterModelBundle'] }) => {
  const { installBundle, getModelsToInstall } = useStarterBundleInstall();
  const { t } = useTranslation();

  const modelsToInstall = useMemo(() => getModelsToInstall(bundle), [bundle, getModelsToInstall]);

  const handleClickBundle = useCallback(() => {
    installBundle(bundle);
  }, [installBundle, bundle]);

  return (
    <Tooltip
      label={
        <Flex flexDir="column" p={1}>
          <Text>{t('modelManager.includesNModels', { n: bundle.models.length })}:</Text>
          <UnorderedList>
            {bundle.models.map((model, index) => (
              <ListItem key={index} wordBreak="break-all">
                {model.name}
              </ListItem>
            ))}
          </UnorderedList>
        </Flex>
      }
    >
      <Button size="sm" onClick={handleClickBundle} py={6} isDisabled={modelsToInstall.install.length === 0}>
        <Flex flexDir="column">
          <Text>{isMainModelBase(bundle.name) ? MODEL_TYPE_SHORT_MAP[bundle.name] : bundle.name}</Text>
          {modelsToInstall.install.length > 0 && (
            <Text fontSize="xs">
              ({bundle.models.length} {t('settings.models')})
            </Text>
          )}
          {modelsToInstall.install.length === 0 && <Text fontSize="xs">{t('common.installed')}</Text>}
        </Flex>
      </Button>
    </Tooltip>
  );
};
