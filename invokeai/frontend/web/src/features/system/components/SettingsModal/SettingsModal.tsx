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
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { VALID_LOG_LEVELS } from 'app/logging/useLogger';
import { LOCALSTORAGE_KEYS, LOCALSTORAGE_PREFIX } from 'app/store/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { systemSelector } from 'features/system/store/systemSelectors';
import {
  SystemState,
  consoleLogLevelChanged,
  setEnableImageDebugging,
  setIsNodesEnabled,
  setShouldConfirmOnDelete,
  shouldAntialiasProgressImageChanged,
  shouldLogToConsoleChanged,
} from 'features/system/store/systemSlice';
import { uiSelector } from 'features/ui/store/uiSelectors';
import {
  setShouldShowAdvancedOptions,
  setShouldShowProgressInViewer,
  setShouldUseCanvasBetaLayout,
  setShouldUseSliders,
} from 'features/ui/store/uiSlice';
import { UIState } from 'features/ui/store/uiTypes';
import { isEqual } from 'lodash-es';
import {
  ChangeEvent,
  PropsWithChildren,
  ReactElement,
  cloneElement,
  useCallback,
  useEffect,
} from 'react';
import { useTranslation } from 'react-i18next';
import { LogLevelName } from 'roarr';
import SettingSwitch from './SettingSwitch';
import SettingsClearIntermediates from './SettingsClearIntermediates';
import SettingsSchedulers from './SettingsSchedulers';

const selector = createSelector(
  [systemSelector, uiSelector],
  (system: SystemState, ui: UIState) => {
    const {
      shouldConfirmOnDelete,
      enableImageDebugging,
      consoleLogLevel,
      shouldLogToConsole,
      shouldAntialiasProgressImage,
      isNodesEnabled,
    } = system;

    const {
      shouldUseCanvasBetaLayout,
      shouldUseSliders,
      shouldShowProgressInViewer,
      shouldShowAdvancedOptions,
    } = ui;

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

  useEffect(() => {
    if (!shouldShowDeveloperSettings) {
      dispatch(shouldLogToConsoleChanged(false));
    }
  }, [shouldShowDeveloperSettings, dispatch]);

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
              </StyledFlex>

              <StyledFlex>
                <Heading size="sm">{t('settings.ui')}</Heading>
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

export const StyledFlex = (props: PropsWithChildren) => {
  return (
    <Flex
      sx={{
        flexDirection: 'column',
        gap: 2,
        p: 4,
        borderRadius: 'base',
        bg: 'base.100',
        _dark: {
          bg: 'base.900',
        },
      }}
    >
      {props.children}
    </Flex>
  );
};
