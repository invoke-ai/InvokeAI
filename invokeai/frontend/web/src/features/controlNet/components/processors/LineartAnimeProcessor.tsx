import { Flex } from '@chakra-ui/react';
import IAISlider from 'common/components/IAISlider';
import { memo, useState } from 'react';

const LineartPreprocessor = () => {
  const [detectResolution, setDetectResolution] = useState(512);
  const [imageResolution, setImageResolution] = useState(512);

  return (
    <Flex sx={{ flexDirection: 'column', gap: 2 }}>
      <IAISlider
        label="Detect Resolution"
        value={detectResolution}
        onChange={setDetectResolution}
        min={0}
        max={4096}
        withInput
      />
      <IAISlider
        label="Image Resolution"
        value={imageResolution}
        onChange={setImageResolution}
        min={0}
        max={4096}
        withInput
      />
    </Flex>
  );
};

export default memo(LineartPreprocessor);
