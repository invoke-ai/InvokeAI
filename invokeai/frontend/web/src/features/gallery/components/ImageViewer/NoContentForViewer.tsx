import { Flex, Image, Text } from '@invoke-ai/ui-library';
import { useHasImages } from 'features/gallery/hooks/useHasImages';
import InvokeSymbol from 'public/assets/images/invoke-symbol-char-lrg.svg';
import { useTranslation } from 'react-i18next';

export const NoContentForViewer = () => {
  const hasImages = useHasImages();
  const { t } = useTranslation();

  //   if (hasImages()) {
  //     return <IAINoContentFallback icon={PiImageBold} label={t('gallery.noImageSelected')} />;
  //   }
  return (
    <Flex flexDir="column" gap={4} alignItems="center" textAlign="center" maxW="600px">
      <Image src={InvokeSymbol} w="8rem" h="8rem" />
      <Text fontSize="md">
        To get started, enter a prompt in the box and click{' '}
        <Text as="span" fontWeight="semibold" fontSize="md">
          Invoke
        </Text>{' '}
        to generate your first image. You can choose to save your images directly to the{' '}
        <Text as="span" fontWeight="semibold" fontSize="md">
          Gallery
        </Text>{' '}
        or edit them on the{' '}
        <Text as="span" fontWeight="semibold" fontSize="md">
          Canvas
        </Text>
        .
      </Text>
      <Text fontSize="md">
        Want more guidance? Check out our{' '}
        <Text
          as="a"
          href="http://youtube.com"
          target="_blank"
          fontWeight="semibold"
          fontSize="md"
          textDecor="underline"
        >
          Getting Started Series
        </Text>{' '}
        for tips on unlocking the full potential of the Invoke Studio.
      </Text>
    </Flex>
  );
};
