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
import { setModel } from '../../../app/socketio/actions';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import IAISelect from '../../../common/components/IAISelect';
import { persistor } from '../../../main';
import {
  setShouldConfirmOnDelete,
  setShouldDisplayGuides,
  setShouldDisplayInProgress,
  SystemState,
} from '../systemSlice';
import SettingsModalItem from './SettingsModalItem';

const systemSelector = createSelector(
  (state: RootState) => state.system,
  (system: SystemState) => {
    const {
      shouldDisplayInProgress,
      shouldConfirmOnDelete,
      shouldDisplayGuides,
      available_models,
    } = system;
    return {
      shouldDisplayInProgress,
      shouldConfirmOnDelete,
      shouldDisplayGuides,
      models: available_models
        ? _.map(available_models, (model, key) => key)
        : [],
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
    shouldDisplayInProgress,
    shouldConfirmOnDelete,
    shouldDisplayGuides,
    models,
  } = useAppSelector(systemSelector);

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

  const handleChangeModel = (e: ChangeEvent<HTMLSelectElement>) => {
    dispatch(setModel(e.target.value));
  };

  return (
    <>
      {cloneElement(children, {
        onClick: onSettingsModalOpen,
      })}

      <Modal isOpen={isSettingsModalOpen} onClose={onSettingsModalClose}>
        <ModalOverlay />
        <ModalContent className="settings-modal">
          <ModalHeader className="settings-modal-header">Settings</ModalHeader>
          <ModalCloseButton />
          <ModalBody className="settings-modal-content">
            <IAISelect
              label="Model"
              validValues={models}
              onChange={handleChangeModel}
            />
            <div className="settings-modal-items">
              <SettingsModalItem
                settingTitle="Display In-Progress Images (slower)"
                isChecked={shouldDisplayInProgress}
                dispatcher={setShouldDisplayInProgress}
              />

              <SettingsModalItem
                settingTitle="Confirm on Delete"
                isChecked={shouldConfirmOnDelete}
                dispatcher={setShouldConfirmOnDelete}
              />

              <SettingsModalItem
                settingTitle="Display Help Icons"
                isChecked={shouldDisplayGuides}
                dispatcher={setShouldDisplayGuides}
              />
            </div>

            <div className="settings-modal-reset">
              <Heading size={'md'}>Reset Web UI</Heading>
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
              <Button colorScheme="red" onClick={handleClickResetWebUI}>
                Reset Web UI
              </Button>
            </div>
          </ModalBody>

          <ModalFooter>
            <Button onClick={onSettingsModalClose}>Close</Button>
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
