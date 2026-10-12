import type { SystemStyleObject } from '@chakra-ui/react';

import {
  Box,
  chakra,
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
import { useCallback } from 'react';

export const toaster = createToaster({
  offsets: { bottom: '1rem', left: '1rem', right: '1rem', top: '1rem' },
  placement: 'bottom-end',
  pauseOnPageIdle: true,
});

/** A button a toast offers; choosing it dismisses the toast first. The first action is the primary one. */
export interface ToastAction {
  label: string;
  onClick: () => void;
}

type ChakraToastOptions = Parameters<typeof toaster.create>[0];

/** Toast options with buttons. Chakra's own options name at most one action and type `meta` loosely. */
export type ActionToastOptions = Omit<ChakraToastOptions, 'action' | 'meta'> & { actions: readonly ToastAction[] };

const ACTIONS_META_KEY = 'platformToastActions';

/** Show a toast with buttons; the only writer of the actions `AppToaster` reads back. */
export const createActionToast = ({ actions, ...options }: ActionToastOptions): string =>
  toaster.create({ ...options, meta: { [ACTIONS_META_KEY]: actions } });

const getToastActions = (meta: Record<string, unknown> | undefined): readonly ToastAction[] =>
  (meta?.[ACTIONS_META_KEY] as readonly ToastAction[] | undefined) ?? [];

/**
 * Enter and Space on a focused toast button belong to that button: toasts are not modal, so without this a
 * workbench shortcut bound to the same key (a canvas session's Enter) would take the press first.
 */
export const isToastActivationKeyEvent = (event: KeyboardEvent): boolean =>
  (event.key === 'Enter' || event.key === ' ') &&
  !event.ctrlKey &&
  !event.metaKey &&
  !event.altKey &&
  event.target instanceof Element &&
  event.target.closest('[data-scope="toast"]') !== null;

/**
 * Both buttons share one size and type, and take their fills from the toast's own trigger tokens, so hover and focus
 * read on every status color in both themes. The primary is outlined; the others are not.
 */
const ACTION_CSS: SystemStyleObject = {
  alignItems: 'center',
  bg: 'transparent',
  borderColor: 'transparent',
  borderRadius: 'l2',
  borderWidth: '1px',
  color: 'inherit',
  cursor: 'pointer',
  display: 'inline-flex',
  fontWeight: 'medium',
  height: '7',
  px: '2.5',
  textStyle: 'sm',
  transitionDuration: 'fast',
  transitionProperty: 'background-color',
  whiteSpace: 'nowrap',
  _focusVisible: { outline: '2px solid currentColor', outlineOffset: '1px' },
  _hover: { bg: 'var(--toast-trigger-bg)' },
};
const PRIMARY_ACTION_CSS: SystemStyleObject = {
  ...ACTION_CSS,
  borderColor: 'var(--toast-border-color, {colors.border})',
};

const ToastActionButton = ({
  action,
  isPrimary,
  toastId,
}: {
  action: ToastAction;
  isPrimary: boolean;
  toastId: string | undefined;
}) => {
  const handleClick = useCallback(() => {
    toaster.dismiss(toastId);
    action.onClick();
  }, [action, toastId]);

  return (
    <chakra.button css={isPrimary ? PRIMARY_ACTION_CSS : ACTION_CSS} type="button" onClick={handleClick}>
      {action.label}
    </chakra.button>
  );
};

export const AppToaster = () => (
  <Portal>
    <ChakraToaster toaster={toaster}>
      {(toast) => {
        const actions = getToastActions(toast.meta);

        return (
          <ToastRoot maxW="calc(100vw - 2rem)" w="24rem">
            {/* Unbroken model names, paths, and URLs wrap only once the flex chain may shrink below them. */}
            <HStack align="start" gap="3" minW="0" w="full">
              <ToastIndicator />
              <Stack flex="1" gap="1" minW="0">
                {toast.title ? <ToastTitle>{toast.title}</ToastTitle> : null}
                {toast.description ? <ToastDescription>{toast.description}</ToastDescription> : null}
                {actions.length > 0 ? (
                  <HStack flexWrap="wrap" gap="1.5" pt="1">
                    {actions.map((action, index) => (
                      <ToastActionButton
                        key={action.label}
                        action={action}
                        isPrimary={index === 0}
                        toastId={toast.id}
                      />
                    ))}
                  </HStack>
                ) : null}
              </Stack>
              <ToastCloseTrigger />
            </HStack>
            <Box display="none" />
          </ToastRoot>
        );
      }}
    </ChakraToaster>
  </Portal>
);
