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
        <ToastRoot maxW="calc(100vw - 2rem)" overflowWrap="anywhere" w="24rem">
          <HStack align="start" gap="3">
            <ToastIndicator />
            <Stack gap="1" flex="1">
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
