import { Flex, Text } from '@invoke-ai/ui-library';
// import AnimatedVideoSpinner from 'public/assets/videos/LogoAnim.webm';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import type { Options } from 'react-lottie';
import Lottie from 'react-lottie';

import { LoadingAnimationData } from './LoadingAnimationData';

const options: Options = {
  loop: true,
  autoplay: true,
  animationData: LoadingAnimationData,
  rendererSettings: {
    preserveAspectRatio: 'xMidYMid slice',
  },
};

// This component loads before the theme so we cannot use theme tokens here
const Loading = () => {
  const { t } = useTranslation();
  return (
    <Flex
      position="relative"
      flexDir="column"
      width="100vw"
      height="100vh"
      alignItems="center"
      justifyContent="center"
      bg="#151519"
    >
      <Lottie options={options} width={200} height={200} />
      <Text color="#535a65" fontFamily="sans-serif" fontSize="small" fontWeight={600}>
        {t('common.initializingInvokeAI')}
      </Text>
    </Flex>
  );
};

export default memo(Loading);
