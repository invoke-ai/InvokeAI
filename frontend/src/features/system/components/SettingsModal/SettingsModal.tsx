import {
  Button,
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
import _, { isEqual } from 'lodash';
import { ChangeEvent, cloneElement, ReactElement } from 'react';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import { persistor } from 'main';
import {
  InProgressImageType,
  setEnableImageDebugging,
  setSaveIntermediatesInterval,
  setShouldConfirmOnDelete,
  setShouldDisplayGuides,
  setShouldDisplayInProgressType,
} from 'features/system/store/systemSlice';
import ModelList from './ModelList';
import { IN_PROGRESS_IMAGE_TYPES } from 'app/constants';
import IAISwitch from 'common/components/IAISwitch';
import IAISelect from 'common/components/IAISelect';
import IAINumberInput from 'common/components/IAINumberInput';
import { systemSelector } from 'features/system/store/systemSelectors';
import { optionsSelector } from 'features/options/store/optionsSelectors';

const selector = createSelector(
  [systemSelector, optionsSelector],
  (system) => {
    const {
      shouldDisplayInProgressType,
      shouldConfirmOnDelete,
      shouldDisplayGuides,
      model_list,
      saveIntermediatesInterval,
      enableImageDebugging,
    } = system;

    return {
      shouldDisplayInProgressType,
      shouldConfirmOnDelete,
      shouldDisplayGuides,
      models: _.map(model_list, (_model, key) => key),
      saveIntermediatesInterval,
      enableImageDebugging,
    };
  },
  {
    memoizeOptions: { resultEqualityCheck: isEqual },
  }
);

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

  const steps = useAppSelector((state: RootState) => state.options.steps);

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

      <Modal isOpen={isSettingsModalOpen} onClose={onSettingsModalClose}>
        <ModalOverlay />
        <ModalContent className="modal settings-modal">
          <ModalHeader className="settings-modal-header">Settings</ModalHeader>
          <ModalCloseButton className="modal-close-btn" />
          <ModalBody className="settings-modal-content">
            <div className="settings-modal-items">
              <div className="settings-modal-item">
                <ModelList />
              </div>
              <div
                className="settings-modal-item"
                style={{ gridAutoFlow: 'row', rowGap: '0.5rem' }}
              >
                <IAISelect
                  label={'Display In-Progress Images'}
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
                    label="Save images every n steps"
                    min={1}
                    max={steps}
                    step={1}
                    onChange={handleChangeIntermediateSteps}
                    value={saveIntermediatesInterval}
                    width="auto"
                    textAlign="center"
                  />
                )}
              </div>
              <IAISwitch
                styleClass="settings-modal-item"
                label={'Confirm on Delete'}
                isChecked={shouldConfirmOnDelete}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  dispatch(setShouldConfirmOnDelete(e.target.checked))
                }
              />
              <IAISwitch
                styleClass="settings-modal-item"
                label={'Display Help Icons'}
                isChecked={shouldDisplayGuides}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  dispatch(setShouldDisplayGuides(e.target.checked))
                }
              />
            </div>

            <div className="settings-modal-items">
              <h2 style={{ fontWeight: 'bold' }}>Developer</h2>
              <IAISwitch
                styleClass="settings-modal-item"
                label={'Enable Image Debugging'}
                isChecked={enableImageDebugging}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  dispatch(setEnableImageDebugging(e.target.checked))
                }
              />
            </div>

            <div className="settings-modal-reset">
              <Heading size={'md'}>Reset Web UI</Heading>
              <Button colorScheme="red" onClick={handleClickResetWebUI}>
                Reset Web UI
              </Button>
              <Text>
                Resetting the web UI only resets the browser's local cache of
                your images and remembered settings. It does not delete any
                images from disk.
              </Text>
              <Text>
                If images aren't showing up in the gallery or something else
                isn't working, please try resetting before submitting an issue
                on GitHub.
              </Text>
            </div>
          </ModalBody>

          <ModalFooter>
            <Button onClick={onSettingsModalClose} className="modal-close-btn">
              Close
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>

      <Modal
        closeOnOverlayClick={false}
        isOpen={isRefreshModalOpen}
        onClose={onRefreshModalClose}
        isCentered
      >
        <ModalOverlay bg="blackAlpha.300" backdropFilter="blur(40px)" />
        <ModalContent>
          <ModalBody pb={6} pt={6}>
            <Flex justifyContent={'center'}>
              <Text fontSize={'lg'}>
                Web UI has been reset. Refresh the page to reload.
              </Text>
            </Flex>
          </ModalBody>
        </ModalContent>
      </Modal>
    </>
  );
};

export default SettingsModal;
