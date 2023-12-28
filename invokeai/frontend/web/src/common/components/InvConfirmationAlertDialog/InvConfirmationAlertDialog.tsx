import {
  InvAlertDialog,
  InvAlertDialogBody,
  InvAlertDialogContent,
  InvAlertDialogFooter,
  InvAlertDialogHeader,
  InvAlertDialogOverlay,
} from 'common/components/InvAlertDialog/wrapper';
import { InvButton } from 'common/components/InvButton/InvButton';
import { useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import type { InvConfirmationAlertDialogProps } from './types';

/**
 * This component is a wrapper around InvAlertDialog that provides a confirmation dialog.
 * Its state must be managed externally using chakra's `useDisclosure()` hook.
 */
export const InvConfirmationAlertDialog = (
  props: InvConfirmationAlertDialogProps
) => {
  const { t } = useTranslation();

  const {
    acceptCallback,
    cancelCallback,
    acceptButtonText = t('common.accept'),
    cancelButtonText = t('common.cancel'),
    children,
    title,
    isOpen,
    onClose,
  } = props;

  const cancelRef = useRef<HTMLButtonElement | null>(null);

  const handleAccept = useCallback(() => {
    acceptCallback();
    onClose();
  }, [acceptCallback, onClose]);

  const handleCancel = useCallback(() => {
    cancelCallback && cancelCallback();
    onClose();
  }, [cancelCallback, onClose]);

  return (
    <InvAlertDialog
      isOpen={isOpen}
      leastDestructiveRef={cancelRef}
      onClose={onClose}
      isCentered
    >
      <InvAlertDialogOverlay>
        <InvAlertDialogContent>
          <InvAlertDialogHeader fontSize="lg" fontWeight="bold">
            {title}
          </InvAlertDialogHeader>

          <InvAlertDialogBody>{children}</InvAlertDialogBody>

          <InvAlertDialogFooter>
            <InvButton ref={cancelRef} onClick={handleCancel}>
              {cancelButtonText}
            </InvButton>
            <InvButton colorScheme="error" onClick={handleAccept} ml={3}>
              {acceptButtonText}
            </InvButton>
          </InvAlertDialogFooter>
        </InvAlertDialogContent>
      </InvAlertDialogOverlay>
    </InvAlertDialog>
  );
};
