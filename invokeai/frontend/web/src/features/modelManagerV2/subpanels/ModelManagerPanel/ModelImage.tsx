import { Flex, Icon, Image } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $crossOrigin } from 'app/store/nanostores/authToken';
import { typedMemo } from 'common/util/typedMemo';
import { PiImage } from 'react-icons/pi';

type Props = {
  image_url?: string | null;
};

export const MODEL_IMAGE_THUMBNAIL_SIZE = '40px';
const FALLBACK_ICON_SIZE = '24px';

const ModelImage = ({ image_url }: Props) => {
  const crossOrigin = useStore($crossOrigin);

  if (!image_url) {
    return (
      <Flex
        height={MODEL_IMAGE_THUMBNAIL_SIZE}
        minWidth={MODEL_IMAGE_THUMBNAIL_SIZE}
        borderRadius="base"
        alignItems="center"
        justifyContent="center"
      >
        <Icon color="base.500" as={PiImage} boxSize={FALLBACK_ICON_SIZE} />
      </Flex>
    );
  }

  return (
    <Image
      src={image_url}
      crossOrigin={crossOrigin}
      objectFit="cover"
      objectPosition="50% 50%"
      height={MODEL_IMAGE_THUMBNAIL_SIZE}
      width={MODEL_IMAGE_THUMBNAIL_SIZE}
      minHeight={MODEL_IMAGE_THUMBNAIL_SIZE}
      minWidth={MODEL_IMAGE_THUMBNAIL_SIZE}
      borderRadius="base"
      fallback={
        <Flex
          height={MODEL_IMAGE_THUMBNAIL_SIZE}
          minWidth={MODEL_IMAGE_THUMBNAIL_SIZE}
          borderRadius="base"
          alignItems="center"
          justifyContent="center"
        >
          <Icon color="base.500" as={PiImage} boxSize={FALLBACK_ICON_SIZE} />
        </Flex>
      }
    />
  );
};

export default typedMemo(ModelImage);
