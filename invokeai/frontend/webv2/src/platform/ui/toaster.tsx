import {
  Box,
  HStack,
  Portal,
  Stack,
  ToastCloseTrigger,
  ToastDescription,
  Toaster as ChakraToaster,
  ToastIndicator,
  ToastRoot,
  ToastTitle,
  createToaster,
} from '@chakra-ui/react';

export const toaster = createToaster({
  offsets: { bottom: '1rem', left: '1rem', right: '1rem', top: '1rem' },
  placement: 'bottom-end',
  pauseOnPageIdle: true,
});

export const AppToaster = () => (
  <Portal>
    <ChakraToaster toaster={toaster}>
      {(toast) => (
        <ToastRoot maxW="calc(100vw - 2rem)" w="24rem">
          {/* Unbroken model names, paths, and URLs wrap only once the flex chain may shrink below them. */}
          <HStack align="start" gap="3" minW="0" w="full">
            <ToastIndicator />
            <Stack flex="1" gap="1" minW="0">
              {toast.title ? <ToastTitle>{toast.title}</ToastTitle> : null}
              {toast.description ? <ToastDescription>{toast.description}</ToastDescription> : null}
            </Stack>
            <ToastCloseTrigger />
          </HStack>
          <Box display="none" />
        </ToastRoot>
      )}
    </ChakraToaster>
  </Portal>
);
