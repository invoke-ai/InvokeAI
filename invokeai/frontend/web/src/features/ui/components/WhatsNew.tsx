import { ExternalLink, Flex, ListItem, Text, UnorderedList } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { Trans, useTranslation } from 'react-i18next';

const selectIsLocal = createSelector(selectConfigSlice, (config) => config.isLocal);

export const WhatsNew = () => {
  const { t } = useTranslation();
  const isLocal = useAppSelector(selectIsLocal);

  return (
    <Flex gap={4} flexDir="column">
      <UnorderedList fontSize="sm">
        <ListItem>
          <Trans
            i18nKey="whatsNew.line1"
            components={{
              ItalicComponent: <Text as="span" color="white" fontSize="sm" fontStyle="italic" />,
            }}
          />
        </ListItem>
        <ListItem>{t('whatsNew.line2')}</ListItem>
        <ListItem>{t('whatsNew.line3')}</ListItem>
      </UnorderedList>
      <Flex flexDir="column" gap={1}>
        <ExternalLink
          fontSize="sm"
          fontWeight="semibold"
          label={t('whatsNew.readReleaseNotes')}
          href={
            isLocal
              ? 'https://github.com/invoke-ai/InvokeAI/releases/tag/v5.0.0'
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
