import { Flex, Text, Image } from '@chakra-ui/react';
import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';
import InvokeAILogoImage from 'assets/images/logo.png';
import MediaQuery from 'react-responsive';

const InvokeAILogoComponent = () => {
  const appVersion = useAppSelector(
    (state: RootState) => state.system.app_version
  );

  return (
    <Flex alignItems="center" gap={3} ps={1}>
      <Image src={InvokeAILogoImage} alt="invoke-ai-logo" w="32px" h="32px" />
      <MediaQuery minDeviceWidth={768}>
        <Text fontSize="xl">
          invoke <strong>ai</strong>
        </Text>
        <Text
          sx={{
            fontWeight: 300,
            marginTop: 1,
          }}
          variant="subtext"
        >
          {appVersion}
        </Text>
      </MediaQuery>
    </Flex>
  );
};

export default InvokeAILogoComponent;
