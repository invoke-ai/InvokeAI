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
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import { memo, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

export const [useRefreshAfterResetModal] = buildUseBoolean(false);

const RefreshAfterResetModal = () => {
  useAssertSingleton('RefreshAfterResetModal');
  const { t } = useTranslation();
  const [countdown, setCountdown] = useState(3);

  const refreshModal = useRefreshAfterResetModal();

  useEffect(() => {
    if (!refreshModal.isTrue) {
      return;
    }
    const i = window.setInterval(() => setCountdown((prev) => prev - 1), 1000);
    return () => {
      window.clearInterval(i);
    };
  }, [refreshModal.isTrue]);

  useEffect(() => {
    if (countdown <= 0) {
      window.location.reload();
    }
  }, [countdown]);

  return (
    <>
      <Modal
        closeOnOverlayClick={false}
        isOpen={refreshModal.isTrue}
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
