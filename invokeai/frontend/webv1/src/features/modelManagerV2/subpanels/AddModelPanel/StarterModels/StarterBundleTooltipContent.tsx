import { Flex, ListItem, Text, UnorderedList } from '@invoke-ai/ui-library';
import { useStarterBundleInstallStatus } from 'features/modelManagerV2/hooks/useStarterBundleInstallStatus';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import type { S } from 'services/api/types';

export const StarterBundleTooltipContent = memo(({ bundle }: { bundle: S['StarterModelBundle'] }) => {
  const { t } = useTranslation();
  const { total, install, skip } = useStarterBundleInstallStatus(bundle);

  return (
    <Flex flexDir="column" p={1} gap={2}>
      <Text>{t('modelManager.includesNModels', { n: total })}</Text>
      {install.length === 0 && (
        <Flex flexDir="column">
          <Text fontWeight="semibold">{t('modelManager.allNModelsInstalled', { count: total })}.</Text>
          <UnorderedList>
            {skip.map((model, index) => (
              <ListItem key={index} wordBreak="break-all">
                {model.config.name}
              </ListItem>
            ))}
          </UnorderedList>
        </Flex>
      )}
      {install.length > 0 && (
        <>
          <Flex flexDir="column">
            <Text fontWeight="semibold">{t('modelManager.nToInstall', { count: install.length })}:</Text>
            <UnorderedList>
              {install.map((model, index) => (
                <ListItem key={index} wordBreak="break-all">
                  {model.config.name}
                </ListItem>
              ))}
            </UnorderedList>
          </Flex>
          <Flex flexDir="column">
            <Text fontWeight="semibold">{t('modelManager.nAlreadyInstalled', { count: skip.length })}:</Text>
            <UnorderedList>
              {skip.map((model, index) => (
                <ListItem key={index} wordBreak="break-all">
                  {model.config.name}
                </ListItem>
              ))}
            </UnorderedList>
          </Flex>
        </>
      )}
    </Flex>
  );
});
StarterBundleTooltipContent.displayName = 'StarterBundleTooltipContent';
