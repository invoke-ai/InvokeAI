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
import IAISwitch from 'common/components/IAISwitch';
import { systemSelector } from 'features/system/store/systemSelectors';
import {
  SystemState,
  consoleLogLevelChanged,
  setEnableImageDebugging,
  setShouldConfirmOnDelete,
  setShouldDisplayGuides,
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
import SettingsSchedulers from './SettingsSchedulers';

const selector = createSelector(
  [systemSelector, uiSelector],
  (system: SystemState, ui: UIState) => {
    const {
      shouldConfirmOnDelete,
      shouldDisplayGuides,
      enableImageDebugging,
      consoleLogLevel,
      shouldLogToConsole,
      shouldAntialiasProgressImage,
    } = system;

    const {
      shouldUseCanvasBetaLayout,
      shouldUseSliders,
      shouldShowProgressInViewer,
      shouldShowAdvancedOptions,
    } = ui;

    return {
      shouldConfirmOnDelete,
      shouldDisplayGuides,
      enableImageDebugging,
      shouldUseCanvasBetaLayout,
      shouldUseSliders,
      shouldShowProgressInViewer,
      consoleLogLevel,
      shouldLogToConsole,
      shouldAntialiasProgressImage,
      shouldShowAdvancedOptions,
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
    shouldDisplayGuides,
    enableImageDebugging,
    shouldUseCanvasBetaLayout,
    shouldUseSliders,
    shouldShowProgressInViewer,
    consoleLogLevel,
    shouldLogToConsole,
    shouldAntialiasProgressImage,
    shouldShowAdvancedOptions,
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

  return (
    <>
      {cloneElement(children, {
        onClick: onSettingsModalOpen,
      })}

      <Modal
        isOpen={isSettingsModalOpen}
        onClose={onSettingsModalClose}
        size="xl"
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
                <IAISwitch
                  label={t('settings.confirmOnDelete')}
                  isChecked={shouldConfirmOnDelete}
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    dispatch(setShouldConfirmOnDelete(e.target.checked))
                  }
                />
                {shouldShowAdvancedOptionsSettings && (
                  <IAISwitch
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
                <IAISwitch
                  label={t('settings.displayHelpIcons')}
                  isChecked={shouldDisplayGuides}
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    dispatch(setShouldDisplayGuides(e.target.checked))
                  }
                />
                {shouldShowBetaLayout && (
                  <IAISwitch
                    label={t('settings.useCanvasBeta')}
                    isChecked={shouldUseCanvasBetaLayout}
                    onChange={(e: ChangeEvent<HTMLInputElement>) =>
                      dispatch(setShouldUseCanvasBetaLayout(e.target.checked))
                    }
                  />
                )}
                <IAISwitch
                  label={t('settings.useSlidersForAll')}
                  isChecked={shouldUseSliders}
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    dispatch(setShouldUseSliders(e.target.checked))
                  }
                />
                <IAISwitch
                  label={t('settings.showProgressInViewer')}
                  isChecked={shouldShowProgressInViewer}
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    dispatch(setShouldShowProgressInViewer(e.target.checked))
                  }
                />
                <IAISwitch
                  label={t('settings.antialiasProgressImages')}
                  isChecked={shouldAntialiasProgressImage}
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    dispatch(
                      shouldAntialiasProgressImageChanged(e.target.checked)
                    )
                  }
                />
              </StyledFlex>

              {shouldShowDeveloperSettings && (
                <StyledFlex>
                  <Heading size="sm">{t('settings.developer')}</Heading>
                  <IAISwitch
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
                  <IAISwitch
                    label={t('settings.enableImageDebugging')}
                    isChecked={enableImageDebugging}
                    onChange={(e: ChangeEvent<HTMLInputElement>) =>
                      dispatch(setEnableImageDebugging(e.target.checked))
                    }
                  />
                </StyledFlex>
              )}

              <StyledFlex>
                <Heading size="sm">{t('settings.resetWebUI')}</Heading>
                <IAIButton colorScheme="error" onClick={handleClickResetWebUI}>
                  {t('settings.resetWebUI')}
                </IAIButton>
                {shouldShowResetWebUiText && (
                  <>
                    <Text>{t('settings.resetWebUIDesc1')}</Text>
                    <Text>{t('settings.resetWebUIDesc2')}</Text>
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

const StyledFlex = (props: PropsWithChildren) => {
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
