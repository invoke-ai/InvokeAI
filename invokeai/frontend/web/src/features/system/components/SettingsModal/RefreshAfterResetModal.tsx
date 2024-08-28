import {
  Flex,
  Modal,
  ModalBody,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Text,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import { atom } from 'nanostores';
import { memo, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

const $refreshAfterResetModalState = atom(false);
export const useRefreshAfterResetModal = buildUseBoolean($refreshAfterResetModalState);

const RefreshAfterResetModal = () => {
  const { t } = useTranslation();
  const [countdown, setCountdown] = useState(3);

  const refreshModal = useRefreshAfterResetModal();
  const isOpen = useStore(refreshModal.$boolean);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    const i = window.setInterval(() => setCountdown((prev) => prev - 1), 1000);
    return () => {
      window.clearInterval(i);
    };
  }, [isOpen]);

  useEffect(() => {
    if (countdown <= 0) {
      window.location.reload();
    }
  }, [countdown]);

  return (
    <>
      <Modal
        closeOnOverlayClick={false}
        isOpen={isOpen}
        onClose={refreshModal.setFalse}
        isCentered
        closeOnEsc={false}
        useInert={false}
      >
        <ModalOverlay backdropFilter="blur(40px)" />
        <ModalContent>
          <ModalHeader />
          <ModalBody>
            <Flex justifyContent="center">
              <Text fontSize="lg">
                <Text>
                  {t('settings.resetComplete')} {t('settings.reloadingIn')} {countdown}...
                </Text>
              </Text>
            </Flex>
          </ModalBody>
          <ModalFooter />
        </ModalContent>
      </Modal>
    </>
  );
};

export default memo(RefreshAfterResetModal);
