import splashImageUrl from '@assets/SplashImage.webp';
import { Box, Flex, Heading, HStack, Spinner, Text, VStack } from '@chakra-ui/react';
import { APP_VERSION } from '@platform/runtime/appMetadata';
import { InvokeMark } from '@platform/ui/InvokeMark';
import { useTranslation } from 'react-i18next';

const splashImageStyle = { backgroundPosition: 'center', backgroundSize: 'cover' } as const;

export const WorkbenchSplashScreen = ({ messageKey = 'splash.loadingWorkspace' }: { messageKey?: string }) => {
  const { t } = useTranslation();

  return (
    <Flex align="center" aria-busy="true" h="100vh" justify="center" layerStyle="dotGrid" role="status" w="100vw">
      <Box
        position="relative"
        bg="bg"
        color="fg"
        borderColor="border.subtle"
        borderRadius="3xl"
        borderWidth="1px"
        maxW="2xl"
        p="6"
        shadow="sm"
      >
        <HStack align="stretch" gap="8">
          <Box aspectRatio="1/1" p="14" bgImage={`url(${splashImageUrl})`} rounded="xl" style={splashImageStyle}>
            <InvokeMark size={164} />
          </Box>
          <VStack align="start" flex="1" minH="full" minW="80" textAlign="start" py="2">
            <Heading size="4xl">{t('app.nameWithVersion', { name: t('app.name'), version: APP_VERSION })}</Heading>
            <Text fontSize="xl">{t('splash.tagline')}</Text>
            <Text fontSize="md">{t('splash.artworkBy', { artist: 'Jonathan Pollack' })}</Text>
            <Flex alignItems="center" gap="2" mt="auto">
              <Spinner />
              <Text fontSize="md" flex="1">
                {t(messageKey)}
              </Text>
            </Flex>
            <Text fontSize="xs" mt="auto" fontFamily="mono">
              {t('splash.copyright', { year: new Date().getFullYear() })}
            </Text>
          </VStack>
        </HStack>
      </Box>
    </Flex>
  );
};
