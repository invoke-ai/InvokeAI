import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, Icon, Image } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { PiImage } from 'react-icons/pi';

type Props = {
  image_url?: string | null;
};

export const MODEL_IMAGE_THUMBNAIL_SIZE = '54px';
const FALLBACK_ICON_SIZE = '28px';

const sharedSx: SystemStyleObject = {
  rounded: 'base',
  height: MODEL_IMAGE_THUMBNAIL_SIZE,
  minWidth: MODEL_IMAGE_THUMBNAIL_SIZE,
  bg: 'base.850',
  borderWidth: '1px',
  borderColor: 'base.750',
  borderStyle: 'solid',
};

const ModelImage = ({ image_url }: Props) => {
  if (!image_url) {
    return (
      <Flex alignItems="center" justifyContent="center" sx={sharedSx}>
        <Icon color="base.500" as={PiImage} boxSize={FALLBACK_ICON_SIZE} />
      </Flex>
    );
  }

  return (
    <Image
      src={image_url}
      objectFit="cover"
      objectPosition="50% 50%"
      width={MODEL_IMAGE_THUMBNAIL_SIZE}
      minHeight={MODEL_IMAGE_THUMBNAIL_SIZE}
      sx={sharedSx}
      fallback={
        <Flex
          height={MODEL_IMAGE_THUMBNAIL_SIZE}
          minWidth={MODEL_IMAGE_THUMBNAIL_SIZE}
          sx={sharedSx}
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
