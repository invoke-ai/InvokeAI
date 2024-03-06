import { Box, Image } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';

import { useGetModelConfigQuery } from 'services/api/endpoints/models';

type Props = {
  image_url?: string;
};

const ModelImage = ({ image_url }: Props) => {

  if (!image_url) return <Box height="50px" width="50px" />;

    return (
      <Image
        src={image_url}
        objectFit="cover"
        objectPosition="50% 50%"
        height="50px"
        width="50px"
        borderRadius="base"
      />
    );
};

export default typedMemo(ModelImage);
