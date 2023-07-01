import { Flex, Text, Image } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import InvokeAILogoImage from 'assets/images/logo.png';

const InvokeAILogoComponent = () => {
  const appVersion = useAppSelector(
    (state: RootState) => state.system.app_version
  );

  return (
    <Flex alignItems="center" gap={3} ps={1}>
      <Image
        src={InvokeAILogoImage}
        alt="invoke-ai-logo"
        sx={{
          w: '32px',
          h: '32px',
          minW: '32px',
          minH: '32px',
          userSelect: 'none',
        }}
      />
      <Flex sx={{ gap: 3 }}>
        <Text sx={{ fontSize: 'xl', userSelect: 'none' }}>
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
      </Flex>
    </Flex>
  );
};

export default InvokeAILogoComponent;
