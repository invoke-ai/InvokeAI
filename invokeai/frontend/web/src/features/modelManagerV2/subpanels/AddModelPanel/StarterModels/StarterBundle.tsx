import { Button, Flex, ListItem, Text, Tooltip, UnorderedList } from '@invoke-ai/ui-library';
import { useStarterBundleInstall } from 'features/modelManagerV2/hooks/useStarterBundleInstall';
import { isMainModelBase } from 'features/nodes/types/common';
import { MODEL_TYPE_SHORT_MAP } from 'features/parameters/types/constants';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { StarterModel } from 'services/api/types';

export const StarterBundle = ({ bundleName, bundle }: { bundleName: string; bundle: StarterModel[] }) => {
  const { installBundle, getModelsToInstall } = useStarterBundleInstall();
  const { t } = useTranslation();

  const modelsToInstall = useMemo(() => getModelsToInstall(bundle), [getModelsToInstall, bundle]);

  const handleClickBundle = useCallback(() => {
    installBundle(bundle, bundleName);
  }, [installBundle, bundle, bundleName]);

  return (
    <Tooltip
      label={
        <Flex flexDir="column" p={1}>
          <Text>{t('modelManager.includesNModels', { n: bundle.length })}:</Text>
          <UnorderedList>
            {bundle.map((model, index) => (
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
