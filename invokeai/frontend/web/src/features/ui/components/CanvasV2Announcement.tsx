import { ExternalLink, Flex, ListItem, UnorderedList } from '@invoke-ai/ui-library';
import { useTranslation } from 'react-i18next';

export const CanvasV2Announcement = () => {
  const { t } = useTranslation();
  return (
    <Flex gap={4} flexDir="column">
      <UnorderedList fontSize="sm">
        <ListItem>{t('whatsNew.canvasV2Announcement.newCanvas')}</ListItem>
        <ListItem>{t('whatsNew.canvasV2Announcement.newLayerTypes')}</ListItem>
        <ListItem>{t('whatsNew.canvasV2Announcement.fluxSupport')}</ListItem>
      </UnorderedList>
      <Flex flexDir="column" gap={1}>
        <ExternalLink
          fontSize="sm"
          fontWeight="semibold"
          label={t('whatsNew.canvasV2Announcement.readReleaseNotes')}
          href="https://github.com/invoke-ai/InvokeAI/releases"
        />
        <ExternalLink
          fontSize="sm"
          fontWeight="semibold"
          label={t('whatsNew.canvasV2Announcement.watchReleaseVideo')}
          href="https://www.youtube.com/@invokeai/videos"
        />
        <ExternalLink
          fontSize="sm"
          fontWeight="semibold"
          label={t('whatsNew.canvasV2Announcement.watchUiUpdatesOverview')}
          href="https://www.youtube.com/@invokeai/videos"
        />
      </Flex>
    </Flex>
  );
};
