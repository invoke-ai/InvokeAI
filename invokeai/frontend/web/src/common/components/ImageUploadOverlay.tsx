import { Box, Flex, Heading } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectSelectedBoardId } from 'features/gallery/store/gallerySelectors';
import { selectMaxImageUploadCount } from 'features/system/store/configSlice';
import { memo } from 'react';
import type { DropzoneState } from 'react-dropzone';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { useBoardName } from 'services/api/hooks/useBoardName';

type ImageUploadOverlayProps = {
  dropzone: DropzoneState;
  setIsHandlingUpload: (isHandlingUpload: boolean) => void;
};

const ImageUploadOverlay = (props: ImageUploadOverlayProps) => {
  const { dropzone, setIsHandlingUpload } = props;

  useHotkeys(
    'esc',
    () => {
      setIsHandlingUpload(false);
    },
    [setIsHandlingUpload]
  );

  return (
    <Box position="absolute" top={0} right={0} bottom={0} left={0} zIndex={999} backdropFilter="blur(20px)">
      <Flex position="absolute" top={0} right={0} bottom={0} left={0} bg="base.900" opacity={0.7} />
      <Flex
        position="absolute"
        flexDir="column"
        gap={4}
        top={2}
        right={2}
        bottom={2}
        left={2}
        opacity={1}
        borderWidth={2}
        borderColor={dropzone.isDragAccept ? 'invokeYellow.300' : 'error.500'}
        borderRadius="base"
        borderStyle="dashed"
        transitionProperty="common"
        transitionDuration="0.1s"
        alignItems="center"
        justifyContent="center"
        color={dropzone.isDragReject ? 'error.300' : undefined}
      >
        {dropzone.isDragAccept && <DragAcceptMessage />}
        {!dropzone.isDragAccept && <DragRejectMessage />}
      </Flex>
    </Box>
  );
};
export default memo(ImageUploadOverlay);

const DragAcceptMessage = () => {
  const { t } = useTranslation();
  const selectedBoardId = useAppSelector(selectSelectedBoardId);
  const boardName = useBoardName(selectedBoardId);

  return (
    <>
      <Heading size="lg">{t('gallery.dropToUpload')}</Heading>
      <Heading size="md">{t('toast.imagesWillBeAddedTo', { boardName })}</Heading>
    </>
  );
};

const DragRejectMessage = () => {
  const { t } = useTranslation();
  const maxImageUploadCount = useAppSelector(selectMaxImageUploadCount);

  if (maxImageUploadCount === undefined) {
    return (
      <>
        <Heading size="lg">{t('toast.invalidUpload')}</Heading>
        <Heading size="md">{t('toast.uploadFailedInvalidUploadDesc')}</Heading>
      </>
    );
  }

  return (
    <>
      <Heading size="lg">{t('toast.invalidUpload')}</Heading>
      <Heading size="md">{t('toast.uploadFailedInvalidUploadDesc_withCount', { count: maxImageUploadCount })}</Heading>
    </>
  );
};
