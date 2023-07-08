import { Flex, Image, Text } from '@chakra-ui/react';
import InvokeAILogoImage from 'assets/images/logo.png';
import { useGetAppVersionQuery } from 'services/api/endpoints/appInfo';

const InvokeAILogoComponent = () => {
  const { data: appVersion } = useGetAppVersionQuery();

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
      <Flex sx={{ gap: 3, alignItems: 'center' }}>
        <Text sx={{ fontSize: 'xl', userSelect: 'none' }}>
          invoke <strong>ai</strong>
        </Text>
        {appVersion && (
          <Text
            sx={{
              fontWeight: 600,
              marginTop: 1,
              color: 'base.300',
              fontSize: 14,
            }}
            variant="subtext"
          >
            {appVersion.version}
          </Text>
        )}
      </Flex>
    </Flex>
  );
};

export default InvokeAILogoComponent;
