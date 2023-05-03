import { Flex, Icon, Image } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';
import { isEqual } from 'lodash-es';
import { memo } from 'react';
import { FaImage } from 'react-icons/fa';

const selector = createSelector(
  [systemSelector],
  (system) => {
    const { progressImage } = system;

    return {
      progressImage,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const ProgressImage = () => {
  const { progressImage } = useAppSelector(selector);
  return progressImage ? (
    <Flex
      sx={{
        position: 'relative',
        w: 'full',
        h: 'full',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Image
        draggable={false}
        src={progressImage.dataURL}
        width={progressImage.width}
        height={progressImage.height}
        sx={{
          position: 'absolute',
          objectFit: 'contain',
          maxWidth: '100%',
          maxHeight: '100%',
          height: 'auto',
          borderRadius: 'base',
          p: 2,
        }}
      />
    </Flex>
  ) : (
    <Flex
      sx={{
        w: 'full',
        h: 'full',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Icon color="base.400" boxSize={32} as={FaImage} />
    </Flex>
  );
};

export default memo(ProgressImage);
