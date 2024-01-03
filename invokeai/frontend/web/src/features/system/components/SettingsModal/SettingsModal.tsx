import { Flex, useDisclosure } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvHeading } from 'common/components/InvHeading/wrapper';
import {
  InvModal,
  InvModalBody,
  InvModalCloseButton,
  InvModalContent,
  InvModalFooter,
  InvModalHeader,
  InvModalOverlay,
} from 'common/components/InvModal/wrapper';
import { InvSwitch } from 'common/components/InvSwitch/wrapper';
import { InvText } from 'common/components/InvText/wrapper';
import { useClearStorage } from 'common/hooks/useClearStorage';
import { shouldUseCpuNoiseChanged } from 'features/parameters/store/generationSlice';
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

import SettingsClearIntermediates from './SettingsClearIntermediates';
import { SettingsLanguageSelect } from './SettingsLanguageSelect';
import { SettingsLogLevelSelect } from './SettingsLogLevelSelect';
import StyledFlex from './StyledFlex';

const selector = createMemoizedSelector(
  [stateSelector],
  ({ system, ui, generation }) => {
    const {
      shouldConfirmOnDelete,
      enableImageDebugging,
      shouldLogToConsole,
      shouldAntialiasProgressImage,
      shouldUseNSFWChecker,
      shouldUseWatermarker,
      shouldEnableInformationalPopovers,
    } = system;
    const { shouldUseCpuNoise } = generation;
    const { shouldShowProgressInViewer } = ui;

    return {
      shouldUseCpuNoise,
      shouldConfirmOnDelete,
      enableImageDebugging,
      shouldShowProgressInViewer,
      shouldLogToConsole,
      shouldAntialiasProgressImage,
      shouldUseNSFWChecker,
      shouldUseWatermarker,
      shouldEnableInformationalPopovers,
    };
  }
);

type ConfigOptions = {
  shouldShowDeveloperSettings: boolean;
  shouldShowResetWebUiText: boolean;
  shouldShowAdvancedOptionsSettings: boolean;
  shouldShowClearIntermediates: boolean;
  shouldShowLocalizationToggle: boolean;
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

  const shouldShowDeveloperSettings =
    config?.shouldShowDeveloperSettings ?? true;
  const shouldShowResetWebUiText = config?.shouldShowResetWebUiText ?? true;
  const shouldShowClearIntermediates =
    config?.shouldShowClearIntermediates ?? true;
  const shouldShowLocalizationToggle =
    config?.shouldShowLocalizationToggle ?? true;

  useEffect(() => {
    if (!shouldShowDeveloperSettings) {
      dispatch(shouldLogToConsoleChanged(false));
    }
  }, [shouldShowDeveloperSettings, dispatch]);

  const { isNSFWCheckerAvailable, isWatermarkerAvailable } =
    useGetAppConfigQuery(undefined, {
      selectFromResult: ({ data }) => ({
        isNSFWCheckerAvailable:
          data?.nsfw_methods.includes('nsfw_checker') ?? false,
        isWatermarkerAvailable:
          data?.watermarking_methods.includes('invisible_watermark') ?? false,
      }),
    });

  const {
    isOpen: isSettingsModalOpen,
    onOpen: onSettingsModalOpen,
    onClose: onSettingsModalClose,
  } = useDisclosure();

  const {
    isOpen: isRefreshModalOpen,
    onOpen: onRefreshModalOpen,
    onClose: onRefreshModalClose,
  } = useDisclosure();

  const {
    shouldUseCpuNoise,
    shouldConfirmOnDelete,
    enableImageDebugging,
    shouldShowProgressInViewer,
    shouldLogToConsole,
    shouldAntialiasProgressImage,
    shouldUseNSFWChecker,
    shouldUseWatermarker,
    shouldEnableInformationalPopovers,
  } = useAppSelector(selector);

  const clearStorage = useClearStorage();

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
        onClick: onSettingsModalOpen,
      })}

      <InvModal
        isOpen={isSettingsModalOpen}
        onClose={onSettingsModalClose}
        size="2xl"
        isCentered
      >
        <InvModalOverlay />
        <InvModalContent>
          <InvModalHeader bg="none">{t('common.settingsLabel')}</InvModalHeader>
          <InvModalCloseButton />
          <InvModalBody>
            <Flex gap={4} flexDir="column">
              <StyledFlex>
                <InvHeading size="sm">{t('settings.general')}</InvHeading>
                <InvControl label={t('settings.confirmOnDelete')}>
                  <InvSwitch
                    isChecked={shouldConfirmOnDelete}
                    onChange={handleChangeShouldConfirmOnDelete}
                  />
                </InvControl>
              </StyledFlex>

              <StyledFlex>
                <InvHeading size="sm">{t('settings.generation')}</InvHeading>
                <InvControl
                  label={t('settings.enableNSFWChecker')}
                  isDisabled={!isNSFWCheckerAvailable}
                >
                  <InvSwitch
                    isChecked={shouldUseNSFWChecker}
                    onChange={handleChangeShouldUseNSFWChecker}
                  />
                </InvControl>
                <InvControl
                  label={t('settings.enableInvisibleWatermark')}
                  isDisabled={!isWatermarkerAvailable}
                >
                  <InvSwitch
                    isChecked={shouldUseWatermarker}
                    onChange={handleChangeShouldUseWatermarker}
                  />
                </InvControl>
              </StyledFlex>

              <StyledFlex>
                <InvHeading size="sm">{t('settings.ui')}</InvHeading>
                <InvControl label={t('settings.showProgressInViewer')}>
                  <InvSwitch
                    isChecked={shouldShowProgressInViewer}
                    onChange={handleChangeShouldShowProgressInViewer}
                  />
                </InvControl>
                <InvControl label={t('settings.antialiasProgressImages')}>
                  <InvSwitch
                    isChecked={shouldAntialiasProgressImage}
                    onChange={handleChangeShouldAntialiasProgressImage}
                  />
                </InvControl>
                <InvControl
                  label={t('parameters.useCpuNoise')}
                  feature="noiseUseCPU"
                >
                  <InvSwitch
                    isChecked={shouldUseCpuNoise}
                    onChange={handleChangeShouldUseCpuNoise}
                  />
                </InvControl>
                {shouldShowLocalizationToggle && <SettingsLanguageSelect />}
                <InvControl label={t('settings.enableInformationalPopovers')}>
                  <InvSwitch
                    isChecked={shouldEnableInformationalPopovers}
                    onChange={handleChangeShouldEnableInformationalPopovers}
                  />
                </InvControl>
              </StyledFlex>

              {shouldShowDeveloperSettings && (
                <StyledFlex>
                  <InvHeading size="sm">{t('settings.developer')}</InvHeading>
                  <InvControl label={t('settings.shouldLogToConsole')}>
                    <InvSwitch
                      isChecked={shouldLogToConsole}
                      onChange={handleLogToConsoleChanged}
                    />
                  </InvControl>
                  <SettingsLogLevelSelect />
                  <InvControl label={t('settings.enableImageDebugging')}>
                    <InvSwitch
                      isChecked={enableImageDebugging}
                      onChange={handleChangeEnableImageDebugging}
                    />
                  </InvControl>
                </StyledFlex>
              )}

              {shouldShowClearIntermediates && <SettingsClearIntermediates />}

              <StyledFlex>
                <InvHeading size="sm">{t('settings.resetWebUI')}</InvHeading>
                <InvButton colorScheme="error" onClick={handleClickResetWebUI}>
                  {t('settings.resetWebUI')}
                </InvButton>
                {shouldShowResetWebUiText && (
                  <>
                    <InvText variant="subtext">
                      {t('settings.resetWebUIDesc1')}
                    </InvText>
                    <InvText variant="subtext">
                      {t('settings.resetWebUIDesc2')}
                    </InvText>
                  </>
                )}
              </StyledFlex>
            </Flex>
          </InvModalBody>

          <InvModalFooter>
            <InvButton onClick={onSettingsModalClose}>
              {t('common.close')}
            </InvButton>
          </InvModalFooter>
        </InvModalContent>
      </InvModal>

      <InvModal
        closeOnOverlayClick={false}
        isOpen={isRefreshModalOpen}
        onClose={onRefreshModalClose}
        isCentered
        closeOnEsc={false}
      >
        <InvModalOverlay backdropFilter="blur(40px)" />
        <InvModalContent>
          <InvModalHeader />
          <InvModalBody>
            <Flex justifyContent="center">
              <InvText fontSize="lg">
                <InvText>
                  {t('settings.resetComplete')} {t('settings.reloadingIn')}{' '}
                  {countdown}...
                </InvText>
              </InvText>
            </Flex>
          </InvModalBody>
          <InvModalFooter />
        </InvModalContent>
      </InvModal>
    </>
  );
};

export default memo(SettingsModal);
