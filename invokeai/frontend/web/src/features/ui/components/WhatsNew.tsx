import { ExternalLink, Flex, ListItem, Text, UnorderedList } from '@invoke-ai/ui-library';
import type { ReactNode } from 'react';
import { useMemo } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { useGetAppVersionQuery } from 'services/api/endpoints/appInfo';

const components = {
  StrongComponent: <Text as="span" color="white" fontSize="sm" fontWeight="semibold" />,
};

export const WhatsNew = () => {
  const { t } = useTranslation();
  const { data } = useGetAppVersionQuery();

  const items = useMemo<ReactNode[]>(() => {
    const tKeys = t<string, { returnObjects: true }, string[]>('whatsNew.items', {
      returnObjects: true,
    });

    return tKeys.map((key, index) => (
      <ListItem key={`${key}-${index}`}>
        <Trans i18nKey={key} components={components} />
      </ListItem>
    ));
  }, [t]);

  return (
    <Flex gap={4} flexDir="column">
      <UnorderedList fontSize="sm">{items}</UnorderedList>
      <Flex flexDir="column" gap={1}>
        <ExternalLink
          fontSize="sm"
          fontWeight="semibold"
          label={t('whatsNew.readReleaseNotes')}
          href={`https://github.com/invoke-ai/InvokeAI/releases/tag/v${data?.version}`}
        />
        <ExternalLink
          fontSize="sm"
          fontWeight="semibold"
          label={t('whatsNew.watchRecentReleaseVideos')}
          href="https://www.youtube.com/@invokeai/videos"
        />
      </Flex>
    </Flex>
  );
};
