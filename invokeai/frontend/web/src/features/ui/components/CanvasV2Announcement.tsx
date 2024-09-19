import { Flex, Icon, ListItem, Text, UnorderedList } from '@invoke-ai/ui-library';
import { useTranslation } from 'react-i18next';
import { PiArrowSquareOutBold } from 'react-icons/pi';

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
        <Flex gap={2}>
          <Text as="a" target="_blank" href="https://github.com/invoke-ai/InvokeAI/releases" fontWeight="semibold">
            {t('whatsNew.canvasV2Announcement.readReleaseNotes')}
          </Text>
          <Icon as={PiArrowSquareOutBold} />
        </Flex>
        <Flex gap={2}>
          <Text as="a" target="_blank" href="https://www.youtube.com/@invokeai/videos" fontWeight="semibold">
            {t('whatsNew.canvasV2Announcement.watchReleaseVideo')}
          </Text>
          <Icon as={PiArrowSquareOutBold} />
        </Flex>
        <Flex gap={2}>
          <Text as="a" target="_blank" href="https://www.youtube.com/@invokeai/videos" fontWeight="semibold">
            {t('whatsNew.canvasV2Announcement.watchUiUpdatesOverview')}
          </Text>
          <Icon as={PiArrowSquareOutBold} />
        </Flex>
      </Flex>
    </Flex>
  );
};
