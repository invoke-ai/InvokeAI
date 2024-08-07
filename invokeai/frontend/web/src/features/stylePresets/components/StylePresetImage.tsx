import { Flex, Icon, Image } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { PiImage } from 'react-icons/pi';

const IMAGE_THUMBNAIL_SIZE = '40px';
const FALLBACK_ICON_SIZE = '24px';

const StylePresetImage = ({ presetImageUrl }: { presetImageUrl: string | null }) => {
  return (
    <Image
      src={presetImageUrl || ''}
      fallbackStrategy="beforeLoadOrError"
      fallback={
        <Flex
          height={IMAGE_THUMBNAIL_SIZE}
          minWidth={IMAGE_THUMBNAIL_SIZE}
          bg="base.650"
          borderRadius="base"
          alignItems="center"
          justifyContent="center"
        >
          <Icon color="base.500" as={PiImage} boxSize={FALLBACK_ICON_SIZE} />
        </Flex>
      }
      objectFit="cover"
      objectPosition="50% 50%"
      height={IMAGE_THUMBNAIL_SIZE}
      width={IMAGE_THUMBNAIL_SIZE}
      minHeight={IMAGE_THUMBNAIL_SIZE}
      minWidth={IMAGE_THUMBNAIL_SIZE}
      borderRadius="base"
    />
  );
};

export default typedMemo(StylePresetImage);
