import { Box, Flex, Heading } from '@invoke-ai/ui-library';
import type { AnimationProps } from 'framer-motion';
import { motion } from 'framer-motion';
import { memo } from 'react';
import type { DropzoneState } from 'react-dropzone';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

const initial: AnimationProps['initial'] = {
  opacity: 0,
};
const animate: AnimationProps['animate'] = {
  opacity: 1,
  transition: { duration: 0.1 },
};
const exit: AnimationProps['exit'] = {
  opacity: 0,
  transition: { duration: 0.1 },
};

type ImageUploadOverlayProps = {
  dropzone: DropzoneState;
  setIsHandlingUpload: (isHandlingUpload: boolean) => void;
};

const ImageUploadOverlay = (props: ImageUploadOverlayProps) => {
  const { t } = useTranslation();
  const { dropzone, setIsHandlingUpload } = props;

  useHotkeys(
    'esc',
    () => {
      setIsHandlingUpload(false);
    },
    [setIsHandlingUpload]
  );

  return (
    <Box
      key="image-upload-overlay"
      initial={initial}
      animate={animate}
      exit={exit}
      as={motion.div}
      position="absolute"
      top={0}
      insetInlineStart={0}
      width="100vw"
      height="100vh"
      zIndex={999}
      backdropFilter="blur(20px)"
    >
      <Flex
        position="absolute"
        top={0}
        insetInlineStart={0}
        w="full"
        h="full"
        bg="base.900"
        opacity={0.7}
        alignItems="center"
        justifyContent="center"
        transitionProperty="common"
        transitionDuration="0.1s"
      />
      <Flex
        position="absolute"
        top={0}
        insetInlineStart={0}
        width="full"
        height="full"
        alignItems="center"
        justifyContent="center"
        p={4}
      >
        <Flex
          width="full"
          height="full"
          alignItems="center"
          justifyContent="center"
          flexDir="column"
          gap={4}
          borderWidth={3}
          borderRadius="xl"
          borderStyle="dashed"
          color="base.100"
          borderColor="base.200"
        >
          {dropzone.isDragAccept ? (
            <Heading size="lg">{t('gallery.dropToUpload')}</Heading>
          ) : (
            <>
              <Heading size="lg">{t('toast.invalidUpload')}</Heading>
              <Heading size="md">{t('toast.uploadFailedInvalidUploadDesc')}</Heading>
            </>
          )}
        </Flex>
      </Flex>
    </Box>
  );
};
export default memo(ImageUploadOverlay);
