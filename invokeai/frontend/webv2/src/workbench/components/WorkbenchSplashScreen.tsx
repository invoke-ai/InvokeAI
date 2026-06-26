import splashImageUrl from '@assets/SplashImage.webp';
import { Box, Flex, Heading, HStack, Spinner, Text, VStack } from '@chakra-ui/react';
import { InvokeMark } from '@workbench/components/InvokeMark';

const splashImageStyle = { backgroundPosition: 'center', backgroundSize: 'cover' } as const;

export const WorkbenchSplashScreen = ({ message = 'Loading workspace' }: { message?: string }) => (
  <Flex align="center" aria-busy="true" bg="bg" color="fg" h="100vh" justify="center" role="status" w="100vw">
    <Box
      position="relative"
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
          <Heading size="2xl">Invoke AI 7.0</Heading>
          <Text fontSize="md">Image Generation for Creatives</Text>
          <Text fontSize="xs">Artwork by John Smith</Text>
          <Flex alignItems="center" gap="2" mt="auto">
            <Spinner size="xs" />
            <Text fontSize="xs" flex="1">
              {message}
            </Text>
          </Flex>
          <Text fontSize="2xs" mt="auto" fontFamily="mono">
            &copy; {new Date().getFullYear()} Invoke AI | All rights reserved
          </Text>
        </VStack>
      </HStack>
    </Box>
  </Flex>
);
