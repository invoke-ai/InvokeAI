import {
  Flex,
  Heading,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Text,
  useColorMode,
  useDisclosure,
} from '@chakra-ui/react';
import { VALID_LOG_LEVELS } from 'app/logging/logger';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { useClearStorage } from 'common/hooks/useClearStorage';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { languageSelector } from 'features/system/store/systemSelectors';
import {
  consoleLogLevelChanged,
  languageChanged,
  setEnableImageDebugging,
  setShouldConfirmOnDelete,
  setShouldEnableInformationalPopovers,
  shouldAntialiasProgressImageChanged,
  shouldLogToConsoleChanged,
  shouldUseNSFWCheckerChanged,
  shouldUseWatermarkerChanged,
} from 'features/system/store/systemSlice';
import { LANGUAGES } from 'features/system/store/types';
import {
  setShouldAutoChangeDimensions,
  setShouldShowProgressInViewer,
  setShouldUseSliders,
} from 'features/ui/store/uiSlice';
import { isEqual } from 'lodash-es';
import {
  ChangeEvent,
  ReactElement,
  cloneElement,
  memo,
  useCallback,
  useEffect,
  useState,
} from 'react';
import { useTranslation } from 'react-i18next';
import { LogLevelName } from 'roarr';
import { useGetAppConfigQuery } from 'services/api/endpoints/appInfo';
import SettingSwitch from './SettingSwitch';
import SettingsClearIntermediates from './SettingsClearIntermediates';
import SettingsSchedulers from './SettingsSchedulers';
import StyledFlex from './StyledFlex';

const selector = createMemoizedSelector(
  [stateSelector],
  ({ system, ui }) => {
    const {
      shouldConfirmOnDelete,
      enableImageDebugging,
      consoleLogLevel,
      shouldLogToConsole,
      shouldAntialiasProgressImage,
      shouldUseNSFWChecker,
      shouldUseWatermarker,
      shouldEnableInformationalPopovers,
    } = system;

    const {
      shouldUseSliders,
      shouldShowProgressInViewer,
      shouldAutoChangeDimensions,
    } = ui;

    return {
      shouldConfirmOnDelete,
      enableImageDebugging,
      shouldUseSliders,
      shouldShowProgressInViewer,
      consoleLogLevel,
      shouldLogToConsole,
      shouldAntialiasProgressImage,
      shouldUseNSFWChecker,
      shouldUseWatermarker,
      shouldAutoChangeDimensions,
      shouldEnableInformationalPopovers,
    };
  },
  {
    memoizeOptions: { resultEqualityCheck: isEqual },
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
    shouldConfirmOnDelete,
    enableImageDebugging,
    shouldUseSliders,
    shouldShowProgressInViewer,
    consoleLogLevel,
    shouldLogToConsole,
    shouldAntialiasProgressImage,
    shouldUseNSFWChecker,
    shouldUseWatermarker,
    shouldAutoChangeDimensions,
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

  const handleLogLevelChanged = useCallback(
    (v: string) => {
      dispatch(consoleLogLevelChanged(v as LogLevelName));
    },
    [dispatch]
  );

  const handleLanguageChanged = useCallback(
    (l: string) => {
      dispatch(languageChanged(l as keyof typeof LANGUAGES));
    },
    [dispatch]
  );

  const handleLogToConsoleChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldLogToConsoleChanged(e.target.checked));
    },
    [dispatch]
  );

  const { colorMode, toggleColorMode } = useColorMode();

  const isLocalizationEnabled =
    useFeatureStatus('localization').isFeatureEnabled;
  const language = useAppSelector(languageSelector);

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
  const handleChangeShouldUseSliders = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setShouldUseSliders(e.target.checked));
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
  const handleChangeShouldAutoChangeDimensions = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setShouldAutoChangeDimensions(e.target.checked));
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

  return (
    <>
      {cloneElement(children, {
        onClick: onSettingsModalOpen,
      })}

      <Modal
        isOpen={isSettingsModalOpen}
        onClose={onSettingsModalClose}
        size="2xl"
        isCentered
      >
        <ModalOverlay />
        <ModalContent>
          <ModalHeader bg="none">{t('common.settingsLabel')}</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <Flex sx={{ gap: 4, flexDirection: 'column' }}>
              <StyledFlex>
                <Heading size="sm">{t('settings.general')}</Heading>
                <SettingSwitch
                  label={t('settings.confirmOnDelete')}
                  isChecked={shouldConfirmOnDelete}
                  onChange={handleChangeShouldConfirmOnDelete}
                />
              </StyledFlex>

              <StyledFlex>
                <Heading size="sm">{t('settings.generation')}</Heading>
                <SettingsSchedulers />
                <SettingSwitch
                  label={t('settings.enableNSFWChecker')}
                  isDisabled={!isNSFWCheckerAvailable}
                  isChecked={shouldUseNSFWChecker}
                  onChange={handleChangeShouldUseNSFWChecker}
                />
                <SettingSwitch
                  label={t('settings.enableInvisibleWatermark')}
                  isDisabled={!isWatermarkerAvailable}
                  isChecked={shouldUseWatermarker}
                  onChange={handleChangeShouldUseWatermarker}
                />
              </StyledFlex>

              <StyledFlex>
                <Heading size="sm">{t('settings.ui')}</Heading>
                <SettingSwitch
                  label={t('common.darkMode')}
                  isChecked={colorMode === 'dark'}
                  onChange={toggleColorMode}
                />
                <SettingSwitch
                  label={t('settings.useSlidersForAll')}
                  isChecked={shouldUseSliders}
                  onChange={handleChangeShouldUseSliders}
                />
                <SettingSwitch
                  label={t('settings.showProgressInViewer')}
                  isChecked={shouldShowProgressInViewer}
                  onChange={handleChangeShouldShowProgressInViewer}
                />
                <SettingSwitch
                  label={t('settings.antialiasProgressImages')}
                  isChecked={shouldAntialiasProgressImage}
                  onChange={handleChangeShouldAntialiasProgressImage}
                />
                <SettingSwitch
                  label={t('settings.autoChangeDimensions')}
                  isChecked={shouldAutoChangeDimensions}
                  onChange={handleChangeShouldAutoChangeDimensions}
                />
                {shouldShowLocalizationToggle && (
                  <IAIMantineSelect
                    disabled={!isLocalizationEnabled}
                    label={t('common.languagePickerLabel')}
                    value={language}
                    data={Object.entries(LANGUAGES).map(([value, label]) => ({
                      value,
                      label,
                    }))}
                    onChange={handleLanguageChanged}
                  />
                )}
                <SettingSwitch
                  label={t('settings.enableInformationalPopovers')}
                  isChecked={shouldEnableInformationalPopovers}
                  onChange={handleChangeShouldEnableInformationalPopovers}
                />
              </StyledFlex>

              {shouldShowDeveloperSettings && (
                <StyledFlex>
                  <Heading size="sm">{t('settings.developer')}</Heading>
                  <SettingSwitch
                    label={t('settings.shouldLogToConsole')}
                    isChecked={shouldLogToConsole}
                    onChange={handleLogToConsoleChanged}
                  />
                  <IAIMantineSelect
                    disabled={!shouldLogToConsole}
                    label={t('settings.consoleLogLevel')}
                    onChange={handleLogLevelChanged}
                    value={consoleLogLevel}
                    data={VALID_LOG_LEVELS.concat()}
                  />
                  <SettingSwitch
                    label={t('settings.enableImageDebugging')}
                    isChecked={enableImageDebugging}
                    onChange={handleChangeEnableImageDebugging}
                  />
                </StyledFlex>
              )}

              {shouldShowClearIntermediates && <SettingsClearIntermediates />}

              <StyledFlex>
                <Heading size="sm">{t('settings.resetWebUI')}</Heading>
                <IAIButton colorScheme="error" onClick={handleClickResetWebUI}>
                  {t('settings.resetWebUI')}
                </IAIButton>
                {shouldShowResetWebUiText && (
                  <>
                    <Text variant="subtext">
                      {t('settings.resetWebUIDesc1')}
                    </Text>
                    <Text variant="subtext">
                      {t('settings.resetWebUIDesc2')}
                    </Text>
                  </>
                )}
              </StyledFlex>
            </Flex>
          </ModalBody>

          <ModalFooter>
            <IAIButton onClick={onSettingsModalClose}>
              {t('common.close')}
            </IAIButton>
          </ModalFooter>
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
                  {t('settings.resetComplete')} {t('settings.reloadingIn')}{' '}
                  {countdown}...
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
