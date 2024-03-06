import { Box, Image } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { useState } from 'react';

import { buildModelsUrl } from 'services/api/endpoints/models';

type Props = {
  model_key: string;
};

const ModelImage = ({ model_key }: Props) => {
  const [image, setImage] = useState<string | undefined>(buildModelsUrl(`i/${model_key}/image`));

  if (!image) return <Box height="50px" width="50px"></Box>;

    return (
      <Image
        onError={() => setImage(undefined)}
        src={image}
        objectFit="cover"
        objectPosition="50% 50%"
        height="50px"
        width="50px"
        borderRadius="base"
      />
    );
};

export default typedMemo(ModelImage);
