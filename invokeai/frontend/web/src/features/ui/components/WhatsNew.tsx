import { ExternalLink, Flex, ListItem, Text, UnorderedList } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { useMemo } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { useGetAppVersionQuery } from 'services/api/endpoints/appInfo';

const selectIsLocal = createSelector(selectConfigSlice, (config) => config.isLocal);

export const WhatsNew = () => {
  const { t } = useTranslation();
  const { data } = useGetAppVersionQuery();
  const isLocal = useAppSelector(selectIsLocal);

  const highlights = useMemo(() => (data?.highlights ? data.highlights : []), [data]);

  return (
    <Flex gap={4} flexDir="column">
      <UnorderedList fontSize="sm">
        {highlights.length ? (
          highlights.map((highlight, index) => <ListItem key={index}>{highlight}</ListItem>)
        ) : (
          <>
            <ListItem>
              <Trans
                i18nKey="whatsNew.line1"
                components={{
                  StrongComponent: <Text as="span" color="white" fontSize="sm" fontWeight="semibold" />,
                }}
              />
            </ListItem>
            <ListItem>
              <Trans
                i18nKey="whatsNew.line2"
                components={{
                  StrongComponent: <Text as="span" color="white" fontSize="sm" fontWeight="semibold" />,
                }}
              />
            </ListItem>
          </>
        )}
      </UnorderedList>
      <Flex flexDir="column" gap={1}>
        <ExternalLink
          fontSize="sm"
          fontWeight="semibold"
          label={t('whatsNew.readReleaseNotes')}
          href={
            isLocal
              ? `https://github.com/invoke-ai/InvokeAI/releases/tag/v${data?.version}`
              : 'https://support.invoke.ai/support/solutions/articles/151000178246'
          }
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
