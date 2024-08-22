import { Flex, Icon, Image, Tooltip } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { PiImage } from 'react-icons/pi';

const IMAGE_THUMBNAIL_SIZE = '40px';
const FALLBACK_ICON_SIZE = '24px';

const StylePresetImage = ({ presetImageUrl, imageWidth }: { presetImageUrl: string | null; imageWidth?: number }) => {
  return (
    <Tooltip
      label={
        presetImageUrl && (
          <Image
            src={presetImageUrl}
            draggable={false}
            objectFit="cover"
            maxW={150}
            aspectRatio="1/1"
            borderRadius="base"
            borderBottomRadius="lg"
          />
        )
      }
      p={2}
    >
      <Image
        src={presetImageUrl || ''}
        fallbackStrategy="beforeLoadOrError"
        fallback={
          <Flex
            height={imageWidth || IMAGE_THUMBNAIL_SIZE}
            minWidth={imageWidth || IMAGE_THUMBNAIL_SIZE}
            bg="base.650"
            borderRadius="base"
            alignItems="center"
            justifyContent="center"
          >
            <Icon color="base.500" as={PiImage} boxSize={imageWidth ? imageWidth / 2 : FALLBACK_ICON_SIZE} />
          </Flex>
        }
        objectFit="cover"
        objectPosition="50% 50%"
        height={imageWidth || IMAGE_THUMBNAIL_SIZE}
        width={imageWidth || IMAGE_THUMBNAIL_SIZE}
        minHeight={imageWidth || IMAGE_THUMBNAIL_SIZE}
        minWidth={imageWidth || IMAGE_THUMBNAIL_SIZE}
        borderRadius="base"
      />
    </Tooltip>
  );
};

export default typedMemo(StylePresetImage);
