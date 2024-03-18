import {
  Button,
  Flex,
  FormControl,
  FormControlGroup,
  FormLabel,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Switch,
  Text,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { useClearStorage } from 'common/hooks/useClearStorage';
import { shouldUseCpuNoiseChanged } from 'features/parameters/store/generationSlice';
import { useClearIntermediates } from 'features/system/components/SettingsModal/useClearIntermediates';
import { StickyScrollable } from 'features/system/components/StickyScrollable';
import {
  setEnableImageDebugging,
  setShouldConfirmOnDelete,
  setShouldEnableInformationalPopovers,
  shouldAntialiasProgressImageChanged,
  shouldLogToConsoleChanged,
  shouldUseNSFWCheckerChanged,
  shouldUseWatermarkerChanged,
} from 'features/system/store/systemSlice';
import { setShouldShowProgressInViewer } from 'features/ui/store/uiSlice';
import type { ChangeEvent, ReactElement } from 'react';
import { cloneElement, memo, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetAppConfigQuery } from 'services/api/endpoints/appInfo';

import { SettingsLanguageSelect } from './SettingsLanguageSelect';
import { SettingsLogLevelSelect } from './SettingsLogLevelSelect';

type ConfigOptions = {
  shouldShowDeveloperSettings?: boolean;
  shouldShowResetWebUiText?: boolean;
  shouldShowClearIntermediates?: boolean;
  shouldShowLocalizationToggle?: boolean;
};

type SettingsModalProps = {
  /* The button to open the Settings Modal */
  children: ReactElement;
  config?: ConfigOptions;
};

const SettingsModal = ({ children, config }: SettingsModalProps) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const [countdown, setCountdown] = useState(3);

  const shouldShowDeveloperSettings = config?.shouldShowDeveloperSettings ?? true;
  const shouldShowResetWebUiText = config?.shouldShowResetWebUiText ?? true;
  const shouldShowClearIntermediates = config?.shouldShowClearIntermediates ?? true;
  const shouldShowLocalizationToggle = config?.shouldShowLocalizationToggle ?? true;

  useEffect(() => {
    if (!shouldShowDeveloperSettings) {
      dispatch(shouldLogToConsoleChanged(false));
    }
  }, [shouldShowDeveloperSettings, dispatch]);

  const { isNSFWCheckerAvailable, isWatermarkerAvailable } = useGetAppConfigQuery(undefined, {
    selectFromResult: ({ data }) => ({
      isNSFWCheckerAvailable: data?.nsfw_methods.includes('nsfw_checker') ?? false,
      isWatermarkerAvailable: data?.watermarking_methods.includes('invisible_watermark') ?? false,
    }),
  });

  const {
    clearIntermediates,
    hasPendingItems,
    intermediatesCount,
    isLoading: isLoadingClearIntermediates,
    refetchIntermediatesCount,
  } = useClearIntermediates(shouldShowClearIntermediates);

  const { isOpen: isSettingsModalOpen, onOpen: _onSettingsModalOpen, onClose: onSettingsModalClose } = useDisclosure();

  const { isOpen: isRefreshModalOpen, onOpen: onRefreshModalOpen, onClose: onRefreshModalClose } = useDisclosure();

  const shouldUseCpuNoise = useAppSelector((s) => s.generation.shouldUseCpuNoise);
  const shouldConfirmOnDelete = useAppSelector((s) => s.system.shouldConfirmOnDelete);
  const enableImageDebugging = useAppSelector((s) => s.system.enableImageDebugging);
  const shouldShowProgressInViewer = useAppSelector((s) => s.ui.shouldShowProgressInViewer);
  const shouldLogToConsole = useAppSelector((s) => s.system.shouldLogToConsole);
  const shouldAntialiasProgressImage = useAppSelector((s) => s.system.shouldAntialiasProgressImage);
  const shouldUseNSFWChecker = useAppSelector((s) => s.system.shouldUseNSFWChecker);
  const shouldUseWatermarker = useAppSelector((s) => s.system.shouldUseWatermarker);
  const shouldEnableInformationalPopovers = useAppSelector((s) => s.system.shouldEnableInformationalPopovers);

  const clearStorage = useClearStorage();

  const handleOpenSettingsModel = useCallback(() => {
    if (shouldShowClearIntermediates) {
      refetchIntermediatesCount();
    }
    _onSettingsModalOpen();
  }, [_onSettingsModalOpen, refetchIntermediatesCount, shouldShowClearIntermediates]);

  const handleClickResetWebUI = useCallback(() => {
    clearStorage();
    onSettingsModalClose();
    onRefreshModalOpen();
    setInterval(() => setCountdown((prev) => prev - 1), 1000);
  }, [clearStorage, onSettingsModalClose, onRefreshModalOpen]);

  useEffect(() => {
    if (countdown <= 0) {
      window.location.reload();
    }
  }, [countdown]);

  const handleLogToConsoleChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldLogToConsoleChanged(e.target.checked));
    },
    [dispatch]
  );

  const handleChangeShouldConfirmOnDelete = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setShouldConfirmOnDelete(e.target.checked));
    },
    [dispatch]
  );
  const handleChangeShouldUseNSFWChecker = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldUseNSFWCheckerChanged(e.target.checked));
    },
    [dispatch]
  );
  const handleChangeShouldUseWatermarker = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldUseWatermarkerChanged(e.target.checked));
    },
    [dispatch]
  );
  const handleChangeShouldShowProgressInViewer = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setShouldShowProgressInViewer(e.target.checked));
    },
    [dispatch]
  );
  const handleChangeShouldAntialiasProgressImage = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldAntialiasProgressImageChanged(e.target.checked));
    },
    [dispatch]
  );
  const handleChangeShouldEnableInformationalPopovers = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setShouldEnableInformationalPopovers(e.target.checked));
    },
    [dispatch]
  );
  const handleChangeEnableImageDebugging = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setEnableImageDebugging(e.target.checked));
    },
    [dispatch]
  );
  const handleChangeShouldUseCpuNoise = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldUseCpuNoiseChanged(e.target.checked));
    },
    [dispatch]
  );

  return (
    <>
      {cloneElement(children, {
        onClick: handleOpenSettingsModel,
      })}

      <Modal isOpen={isSettingsModalOpen} onClose={onSettingsModalClose} size="2xl" isCentered>
        <ModalOverlay />
        <ModalContent maxH="80vh" h="68rem">
          <ModalHeader bg="none">{t('common.settingsLabel')}</ModalHeader>
          <ModalCloseButton />
          <ModalBody display="flex" flexDir="column" gap={4}>
            <ScrollableContent>
              <Flex flexDir="column" gap={4}>
                <FormControlGroup formLabelProps={{ flexGrow: 1 }}>
                  <StickyScrollable title={t('settings.general')}>
                    <FormControl>
                      <FormLabel>{t('settings.confirmOnDelete')}</FormLabel>
                      <Switch isChecked={shouldConfirmOnDelete} onChange={handleChangeShouldConfirmOnDelete} />
                    </FormControl>
                  </StickyScrollable>

                  <StickyScrollable title={t('settings.generation')}>
                    <FormControl isDisabled={!isNSFWCheckerAvailable}>
                      <FormLabel>{t('settings.enableNSFWChecker')}</FormLabel>
                      <Switch isChecked={shouldUseNSFWChecker} onChange={handleChangeShouldUseNSFWChecker} />
                    </FormControl>
                    <FormControl isDisabled={!isWatermarkerAvailable}>
                      <FormLabel>{t('settings.enableInvisibleWatermark')}</FormLabel>
                      <Switch isChecked={shouldUseWatermarker} onChange={handleChangeShouldUseWatermarker} />
                    </FormControl>
                  </StickyScrollable>

                  <StickyScrollable title={t('settings.ui')}>
                    <FormControl>
                      <FormLabel>{t('settings.showProgressInViewer')}</FormLabel>
                      <Switch
                        isChecked={shouldShowProgressInViewer}
                        onChange={handleChangeShouldShowProgressInViewer}
                      />
                    </FormControl>
                    <FormControl>
                      <FormLabel>{t('settings.antialiasProgressImages')}</FormLabel>
                      <Switch
                        isChecked={shouldAntialiasProgressImage}
                        onChange={handleChangeShouldAntialiasProgressImage}
                      />
                    </FormControl>
                    <FormControl>
                      <InformationalPopover feature="noiseUseCPU" inPortal={false}>
                        <FormLabel>{t('parameters.useCpuNoise')}</FormLabel>
                      </InformationalPopover>
                      <Switch isChecked={shouldUseCpuNoise} onChange={handleChangeShouldUseCpuNoise} />
                    </FormControl>
                    {shouldShowLocalizationToggle && <SettingsLanguageSelect />}
                    <FormControl>
                      <FormLabel>{t('settings.enableInformationalPopovers')}</FormLabel>
                      <Switch
                        isChecked={shouldEnableInformationalPopovers}
                        onChange={handleChangeShouldEnableInformationalPopovers}
                      />
                    </FormControl>
                  </StickyScrollable>

                  {shouldShowDeveloperSettings && (
                    <StickyScrollable title={t('settings.developer')}>
                      <FormControl>
                        <FormLabel>{t('settings.shouldLogToConsole')}</FormLabel>
                        <Switch isChecked={shouldLogToConsole} onChange={handleLogToConsoleChanged} />
                      </FormControl>
                      <SettingsLogLevelSelect />
                      <FormControl>
                        <FormLabel>{t('settings.enableImageDebugging')}</FormLabel>
                        <Switch isChecked={enableImageDebugging} onChange={handleChangeEnableImageDebugging} />
                      </FormControl>
                    </StickyScrollable>
                  )}

                  {shouldShowClearIntermediates && (
                    <StickyScrollable title={t('settings.clearIntermediates')}>
                      <Button
                        tooltip={hasPendingItems ? t('settings.clearIntermediatesDisabled') : undefined}
                        colorScheme="warning"
                        onClick={clearIntermediates}
                        isLoading={isLoadingClearIntermediates}
                        isDisabled={!intermediatesCount || hasPendingItems}
                      >
                        {t('settings.clearIntermediatesWithCount', {
                          count: intermediatesCount ?? 0,
                        })}
                      </Button>
                      <Text fontWeight="bold">{t('settings.clearIntermediatesDesc1')}</Text>
                      <Text variant="subtext">{t('settings.clearIntermediatesDesc2')}</Text>
                      <Text variant="subtext">{t('settings.clearIntermediatesDesc3')}</Text>
                    </StickyScrollable>
                  )}

                  <StickyScrollable title={t('settings.resetWebUI')}>
                    <Button colorScheme="error" onClick={handleClickResetWebUI}>
                      {t('settings.resetWebUI')}
                    </Button>
                    {shouldShowResetWebUiText && (
                      <>
                        <Text variant="subtext">{t('settings.resetWebUIDesc1')}</Text>
                        <Text variant="subtext">{t('settings.resetWebUIDesc2')}</Text>
                      </>
                    )}
                  </StickyScrollable>
                </FormControlGroup>
              </Flex>
            </ScrollableContent>
          </ModalBody>

          <ModalFooter />
        </ModalContent>
      </Modal>

      <Modal
        closeOnOverlayClick={false}
        isOpen={isRefreshModalOpen}
        onClose={onRefreshModalClose}
        isCentered
        closeOnEsc={false}
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

export default memo(SettingsModal);
