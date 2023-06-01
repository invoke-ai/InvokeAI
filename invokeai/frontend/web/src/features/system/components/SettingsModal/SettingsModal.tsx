import {
  ChakraProps,
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
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAISelect from 'common/components/IAISelect';
import IAISwitch from 'common/components/IAISwitch';
import { systemSelector } from 'features/system/store/systemSelectors';
import {
  consoleLogLevelChanged,
  setEnableImageDebugging,
  setShouldConfirmOnDelete,
  setShouldDisplayGuides,
  shouldAntialiasProgressImageChanged,
  shouldLogToConsoleChanged,
  SystemState,
} from 'features/system/store/systemSlice';
import { uiSelector } from 'features/ui/store/uiSelectors';
import {
  setShouldShowProgressInViewer,
  setShouldUseCanvasBetaLayout,
  setShouldUseSliders,
} from 'features/ui/store/uiSlice';
import { UIState } from 'features/ui/store/uiTypes';
import { isEqual } from 'lodash-es';
import {
  ChangeEvent,
  cloneElement,
  ReactElement,
  useCallback,
  useEffect,
} from 'react';
import { useTranslation } from 'react-i18next';
import { VALID_LOG_LEVELS } from 'app/logging/useLogger';
import { LogLevelName } from 'roarr';
import { LOCALSTORAGE_KEYS, LOCALSTORAGE_PREFIX } from 'app/store/constants';
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
    };
  },
  {
    memoizeOptions: { resultEqualityCheck: isEqual },
  }
);

const modalSectionStyles: ChakraProps['sx'] = {
  flexDirection: 'column',
  gap: 2,
  p: 4,
  bg: 'base.900',
  borderRadius: 'base',
};

type ConfigOptions = {
  shouldShowDeveloperSettings: boolean;
  shouldShowResetWebUiText: boolean;
  shouldShowBetaLayout: boolean;
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
    (e: ChangeEvent<HTMLSelectElement>) => {
      dispatch(consoleLogLevelChanged(e.target.value as LogLevelName));
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
        <ModalContent paddingInlineEnd={4}>
          <ModalHeader>{t('common.settingsLabel')}</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <Flex sx={{ gap: 4, flexDirection: 'column' }}>
              <Flex sx={modalSectionStyles}>
                <Heading size="sm">{t('settings.general')}</Heading>
                <IAISwitch
                  label={t('settings.confirmOnDelete')}
                  isChecked={shouldConfirmOnDelete}
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    dispatch(setShouldConfirmOnDelete(e.target.checked))
                  }
                />
              </Flex>

              <Flex sx={modalSectionStyles}>
                <Heading size="sm">{t('settings.generation')}</Heading>
                <SettingsSchedulers />
              </Flex>

              <Flex sx={modalSectionStyles}>
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
              </Flex>

              {shouldShowDeveloperSettings && (
                <Flex sx={modalSectionStyles}>
                  <Heading size="sm">{t('settings.developer')}</Heading>
                  <IAISwitch
                    label={t('settings.shouldLogToConsole')}
                    isChecked={shouldLogToConsole}
                    onChange={handleLogToConsoleChanged}
                  />
                  <IAISelect
                    horizontal
                    spaceEvenly
                    isDisabled={!shouldLogToConsole}
                    label={t('settings.consoleLogLevel')}
                    onChange={handleLogLevelChanged}
                    value={consoleLogLevel}
                    validValues={VALID_LOG_LEVELS.concat()}
                  />
                  <IAISwitch
                    label={t('settings.enableImageDebugging')}
                    isChecked={enableImageDebugging}
                    onChange={(e: ChangeEvent<HTMLInputElement>) =>
                      dispatch(setEnableImageDebugging(e.target.checked))
                    }
                  />
                </Flex>
              )}

              <Flex sx={modalSectionStyles}>
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
              </Flex>
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
