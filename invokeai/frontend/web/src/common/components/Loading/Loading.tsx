import { Flex, Image, Spinner } from '@invoke-ai/ui-library';
import InvokeLogoWhite from 'public/assets/images/invoke-symbol-wht-lrg.svg';
import { memo } from 'react';

// This component loads before the theme so we cannot use theme tokens here

const Loading = () => {
  return (
    <Flex
      position="absolute"
      width="100dvw"
      height="100dvh"
      alignItems="center"
      justifyContent="center"
      bg="#151519"
      top={0}
      right={0}
      bottom={0}
      left={0}
    >
      <Image src={InvokeLogoWhite} w="8rem" h="8rem" />
      <Spinner
        label="Loading"
        color="grey"
        position="absolute"
        size="sm"
        width="24px !important"
        height="24px !important"
        right="1.5rem"
        bottom="1.5rem"
      />
    </Flex>
  );
};

export default memo(Loading);
