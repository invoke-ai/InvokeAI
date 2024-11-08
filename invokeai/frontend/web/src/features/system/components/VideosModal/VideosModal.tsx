import {
  Flex,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
} from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { buildUseDisclosure } from 'common/hooks/useBoolean';
import { controlCanvasVideos, gettingStartedVideos } from 'features/system/components/VideosModal/data';
import { VideoCardList } from 'features/system/components/VideosModal/VideoCardList';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const [useVideosModal] = buildUseDisclosure(false);

export const VideosModal = memo(() => {
  const { t } = useTranslation();
  const videosModal = useVideosModal();

  return (
    <Modal isOpen={videosModal.isOpen} onClose={videosModal.close} size="2xl" isCentered useInert={false}>
      <ModalOverlay />
      <ModalContent maxH="80vh" h="80vh">
        <ModalHeader bg="none">{t('supportVideos.supportVideos')}</ModalHeader>
        <ModalCloseButton />
        <ModalBody display="flex" flexDir="column" gap={4}>
          <ScrollableContent>
            <Flex flexDir="column" gap={4}>
              <VideoCardList category={t('supportVideos.gettingStarted')} videos={gettingStartedVideos} />
              <VideoCardList category={t('supportVideos.controlCanvas')} videos={controlCanvasVideos} />
            </Flex>
          </ScrollableContent>
        </ModalBody>
        <ModalFooter />
      </ModalContent>
    </Modal>
  );
});

VideosModal.displayName = 'VideosModal';
