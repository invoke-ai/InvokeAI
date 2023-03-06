import {
  ChakraProps,
  Flex,
  Grid,
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
import { IN_PROGRESS_IMAGE_TYPES } from 'app/constants';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAINumberInput from 'common/components/IAINumberInput';
import IAISelect from 'common/components/IAISelect';
import IAISwitch from 'common/components/IAISwitch';
import { systemSelector } from 'features/system/store/systemSelectors';
import {
  InProgressImageType,
  setEnableImageDebugging,
  setSaveIntermediatesInterval,
  setShouldConfirmOnDelete,
  setShouldDisplayGuides,
  setShouldDisplayInProgressType,
  SystemState,
} from 'features/system/store/systemSlice';
import { uiSelector } from 'features/ui/store/uiSelectors';
import {
  setShouldUseCanvasBetaLayout,
  setShouldUseSliders,
} from 'features/ui/store/uiSlice';
import { UIState } from 'features/ui/store/uiTypes';
import { isEqual, map } from 'lodash';
import { persistor } from 'persistor';
import { ChangeEvent, cloneElement, ReactElement } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [systemSelector, uiSelector],
  (system: SystemState, ui: UIState) => {
    const {
      shouldDisplayInProgressType,
      shouldConfirmOnDelete,
      shouldDisplayGuides,
      model_list,
      saveIntermediatesInterval,
      enableImageDebugging,
    } = system;

    const { shouldUseCanvasBetaLayout, shouldUseSliders } = ui;

    return {
      shouldDisplayInProgressType,
      shouldConfirmOnDelete,
      shouldDisplayGuides,
      models: map(model_list, (_model, key) => key),
      saveIntermediatesInterval,
      enableImageDebugging,
      shouldUseCanvasBetaLayout,
      shouldUseSliders,
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
};

type SettingsModalProps = {
  /* The button to open the Settings Modal */
  children: ReactElement;
};

/**
 * Modal for app settings. Also provides Reset functionality in which the
 * app's localstorage is wiped via redux-persist.
 *
 * Secondary post-reset modal is included here.
 */
const SettingsModal = ({ children }: SettingsModalProps) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const steps = useAppSelector((state: RootState) => state.generation.steps);

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
    shouldDisplayInProgressType,
    shouldConfirmOnDelete,
    shouldDisplayGuides,
    saveIntermediatesInterval,
    enableImageDebugging,
    shouldUseCanvasBetaLayout,
    shouldUseSliders,
  } = useAppSelector(selector);

  /**
   * Resets localstorage, then opens a secondary modal informing user to
   * refresh their browser.
   * */
  const handleClickResetWebUI = () => {
    persistor.purge().then(() => {
      onSettingsModalClose();
      onRefreshModalOpen();
    });
  };

  const handleChangeIntermediateSteps = (value: number) => {
    if (value > steps) value = steps;
    if (value < 1) value = 1;
    dispatch(setSaveIntermediatesInterval(value));
  };

  return (
    <>
      {cloneElement(children, {
        onClick: onSettingsModalOpen,
      })}

      <Modal
        isOpen={isSettingsModalOpen}
        onClose={onSettingsModalClose}
        size="xl"
      >
        <ModalOverlay />
        <ModalContent paddingInlineEnd={4}>
          <ModalHeader>{t('common.settingsLabel')}</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <Grid gap={4}>
              <Flex sx={modalSectionStyles}>
                <IAISelect
                  label={t('settings.displayInProgress')}
                  validValues={IN_PROGRESS_IMAGE_TYPES}
                  value={shouldDisplayInProgressType}
                  onChange={(e: ChangeEvent<HTMLSelectElement>) =>
                    dispatch(
                      setShouldDisplayInProgressType(
                        e.target.value as InProgressImageType
                      )
                    )
                  }
                />
                {shouldDisplayInProgressType === 'full-res' && (
                  <IAINumberInput
                    label={t('settings.saveSteps')}
                    min={1}
                    max={steps}
                    step={1}
                    onChange={handleChangeIntermediateSteps}
                    value={saveIntermediatesInterval}
                    width="auto"
                    textAlign="center"
                  />
                )}
                <IAISwitch
                  label={t('settings.confirmOnDelete')}
                  isChecked={shouldConfirmOnDelete}
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    dispatch(setShouldConfirmOnDelete(e.target.checked))
                  }
                />
                <IAISwitch
                  label={t('settings.displayHelpIcons')}
                  isChecked={shouldDisplayGuides}
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    dispatch(setShouldDisplayGuides(e.target.checked))
                  }
                />
                <IAISwitch
                  label={t('settings.useCanvasBeta')}
                  isChecked={shouldUseCanvasBetaLayout}
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    dispatch(setShouldUseCanvasBetaLayout(e.target.checked))
                  }
                />
                <IAISwitch
                  label={t('settings.useSlidersForAll')}
                  isChecked={shouldUseSliders}
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    dispatch(setShouldUseSliders(e.target.checked))
                  }
                />
              </Flex>

              <Flex sx={modalSectionStyles}>
                <Heading size="sm" style={{ fontWeight: 'bold' }}>
                  Developer
                </Heading>
                <IAISwitch
                  label={t('settings.enableImageDebugging')}
                  isChecked={enableImageDebugging}
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    dispatch(setEnableImageDebugging(e.target.checked))
                  }
                />
              </Flex>

              <Flex sx={modalSectionStyles}>
                <Heading size="sm">{t('settings.resetWebUI')}</Heading>
                <IAIButton colorScheme="error" onClick={handleClickResetWebUI}>
                  {t('settings.resetWebUI')}
                </IAIButton>
                <Text>{t('settings.resetWebUIDesc1')}</Text>
                <Text>{t('settings.resetWebUIDesc2')}</Text>
              </Flex>
            </Grid>
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
