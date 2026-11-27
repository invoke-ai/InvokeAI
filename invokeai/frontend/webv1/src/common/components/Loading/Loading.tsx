import { Flex, Image, Spinner } from '@invoke-ai/ui-library';
import InvokeLogoWhite from 'public/assets/images/invoke-symbol-wht-lrg.svg';
import { memo } from 'react';

// This component loads before the theme so we cannot use theme tokens here

const Loading = () => {
  return (
    <Flex
      position="absolute"
      alignItems="center"
      justifyContent="center"
      bg="hsl(220 12% 10% / 1)" // base.900
      inset={0}
      zIndex={99999}
    >
      <Image src={InvokeLogoWhite} w="8rem" h="8rem" />
      <Spinner
        label="Loading"
        color="hsl(220 12% 68% / 1)" // base.300
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
