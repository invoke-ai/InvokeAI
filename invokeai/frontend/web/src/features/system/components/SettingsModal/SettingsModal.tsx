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
  useDisclosure,
  useColorMode,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { VALID_LOG_LEVELS } from 'app/logging/logger';
import { LOCALSTORAGE_KEYS, LOCALSTORAGE_PREFIX } from 'app/store/constants';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { setShouldShowAdvancedOptions } from 'features/parameters/store/generationSlice';
import {
  consoleLogLevelChanged,
  setEnableImageDebugging,
  setIsNodesEnabled,
  setShouldConfirmOnDelete,
  shouldAntialiasProgressImageChanged,
  shouldLogToConsoleChanged,
  shouldUseNSFWCheckerChanged,
  shouldUseWatermarkerChanged,
} from 'features/system/store/systemSlice';
import {
  setShouldShowProgressInViewer,
  setShouldUseCanvasBetaLayout,
  setShouldUseSliders,
} from 'features/ui/store/uiSlice';
import { isEqual } from 'lodash-es';
import {
  ChangeEvent,
  ReactElement,
  cloneElement,
  useCallback,
  useEffect,
} from 'react';
import { useTranslation } from 'react-i18next';
import { LogLevelName } from 'roarr';
import { useGetAppConfigQuery } from 'services/api/endpoints/appInfo';
import SettingSwitch from './SettingSwitch';
import SettingsClearIntermediates from './SettingsClearIntermediates';
import SettingsSchedulers from './SettingsSchedulers';
import StyledFlex from './StyledFlex';
import { useFeatureStatus } from '../../hooks/useFeatureStatus';
import { LANGUAGES } from '../../store/constants';
import { languageChanged } from '../../store/systemSlice';
import { languageSelector } from '../../store/systemSelectors';

const selector = createSelector(
  [stateSelector],
  ({ system, ui, generation }) => {
    const {
      shouldConfirmOnDelete,
      enableImageDebugging,
      consoleLogLevel,
      shouldLogToConsole,
      shouldAntialiasProgressImage,
      isNodesEnabled,
      shouldUseNSFWChecker,
      shouldUseWatermarker,
    } = system;

    const {
      shouldUseCanvasBetaLayout,
      shouldUseSliders,
      shouldShowProgressInViewer,
    } = ui;

    const { shouldShowAdvancedOptions } = generation;

    return {
      shouldConfirmOnDelete,
      enableImageDebugging,
      shouldUseCanvasBetaLayout,
      shouldUseSliders,
      shouldShowProgressInViewer,
      consoleLogLevel,
      shouldLogToConsole,
      shouldAntialiasProgressImage,
      shouldShowAdvancedOptions,
      isNodesEnabled,
      shouldUseNSFWChecker,
      shouldUseWatermarker,
    };
  },
  {
    memoizeOptions: { resultEqualityCheck: isEqual },
  }
);

type ConfigOptions = {
  shouldShowDeveloperSettings: boolean;
  shouldShowResetWebUiText: boolean;
  shouldShowBetaLayout: boolean;
  shouldShowAdvancedOptionsSettings: boolean;
  shouldShowClearIntermediates: boolean;
  shouldShowNodesToggle: boolean;
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

  const shouldShowBetaLayout = config?.shouldShowBetaLayout ?? true;
  const shouldShowDeveloperSettings =
    config?.shouldShowDeveloperSettings ?? true;
  const shouldShowResetWebUiText = config?.shouldShowResetWebUiText ?? true;
  const shouldShowAdvancedOptionsSettings =
    config?.shouldShowAdvancedOptionsSettings ?? true;
  const shouldShowClearIntermediates =
    config?.shouldShowClearIntermediates ?? true;
  const shouldShowNodesToggle = config?.shouldShowNodesToggle ?? true;
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
    shouldUseCanvasBetaLayout,
    shouldUseSliders,
    shouldShowProgressInViewer,
    consoleLogLevel,
    shouldLogToConsole,
    shouldAntialiasProgressImage,
    shouldShowAdvancedOptions,
    isNodesEnabled,
    shouldUseNSFWChecker,
    shouldUseWatermarker,
  } = useAppSelector(selector);

  const handleClickResetWebUI = useCallback(() => {
    // Only remove our keys
    Object.keys(window.localStorage).forEach((key) => {
      if (
        LOCALSTORAGE_KEYS.includes(key) ||
        key.startsWith(LOCALSTORAGE_PREFIX)
      ) {
        localStorage.removeItem(key);
      }
    });
    onSettingsModalClose();
    onRefreshModalOpen();
  }, [onSettingsModalClose, onRefreshModalOpen]);

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

  const handleToggleNodes = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setIsNodesEnabled(e.target.checked));
    },
    [dispatch]
  );

  const { colorMode, toggleColorMode } = useColorMode();

  const isLocalizationEnabled =
    useFeatureStatus('localization').isFeatureEnabled;
  const language = useAppSelector(languageSelector);

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
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    dispatch(setShouldConfirmOnDelete(e.target.checked))
                  }
                />
                {shouldShowAdvancedOptionsSettings && (
                  <SettingSwitch
                    label={t('settings.showAdvancedOptions')}
                    isChecked={shouldShowAdvancedOptions}
                    onChange={(e: ChangeEvent<HTMLInputElement>) =>
                      dispatch(setShouldShowAdvancedOptions(e.target.checked))
                    }
                  />
                )}
              </StyledFlex>

              <StyledFlex>
                <Heading size="sm">{t('settings.generation')}</Heading>
                <SettingsSchedulers />
                <SettingSwitch
                  label="Enable NSFW Checker"
                  isDisabled={!isNSFWCheckerAvailable}
                  isChecked={shouldUseNSFWChecker}
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    dispatch(shouldUseNSFWCheckerChanged(e.target.checked))
                  }
                />
                <SettingSwitch
                  label="Enable Invisible Watermark"
                  isDisabled={!isWatermarkerAvailable}
                  isChecked={shouldUseWatermarker}
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    dispatch(shouldUseWatermarkerChanged(e.target.checked))
                  }
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
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    dispatch(setShouldUseSliders(e.target.checked))
                  }
                />
                <SettingSwitch
                  label={t('settings.showProgressInViewer')}
                  isChecked={shouldShowProgressInViewer}
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    dispatch(setShouldShowProgressInViewer(e.target.checked))
                  }
                />
                <SettingSwitch
                  label={t('settings.antialiasProgressImages')}
                  isChecked={shouldAntialiasProgressImage}
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    dispatch(
                      shouldAntialiasProgressImageChanged(e.target.checked)
                    )
                  }
                />
                {shouldShowBetaLayout && (
                  <SettingSwitch
                    label={t('settings.alternateCanvasLayout')}
                    useBadge
                    badgeLabel={t('settings.beta')}
                    isChecked={shouldUseCanvasBetaLayout}
                    onChange={(e: ChangeEvent<HTMLInputElement>) =>
                      dispatch(setShouldUseCanvasBetaLayout(e.target.checked))
                    }
                  />
                )}
                {shouldShowNodesToggle && (
                  <SettingSwitch
                    label={t('settings.enableNodesEditor')}
                    useBadge
                    isChecked={isNodesEnabled}
                    onChange={handleToggleNodes}
                  />
                )}
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
                    onChange={(e: ChangeEvent<HTMLInputElement>) =>
                      dispatch(setEnableImageDebugging(e.target.checked))
                    }
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
      >
        <ModalOverlay backdropFilter="blur(40px)" />
        <ModalContent>
          <ModalHeader />
          <ModalBody>
            <Flex justifyContent="center">
              <Text fontSize="lg">
                <Text>{t('settings.resetComplete')}</Text>
              </Text>
            </Flex>
          </ModalBody>
          <ModalFooter />
        </ModalContent>
      </Modal>
    </>
  );
};

export default SettingsModal;
