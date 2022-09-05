import {
    IconButton,
    Modal,
    ModalBody,
    ModalCloseButton,
    ModalContent,
    ModalFooter,
    ModalHeader,
    ModalOverlay,
    useDisclosure,
} from '@chakra-ui/react';
import { MdSettings } from 'react-icons/md';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import {
    setShouldDisplayInProgress,
    setShouldFitToWidthHeight,
} from '../../app/sdSlice';
import { RootState } from '../../app/store';
import SDButton from '../../components/SDButton';
import SDSwitch from '../../components/SDSwitch';

const SettingsModalButton = () => {
    const { isOpen, onOpen, onClose } = useDisclosure();
    const { shouldDisplayInProgress, shouldFitToWidthHeight } = useAppSelector(
        (state: RootState) => state.sd
    );

    const dispatch = useAppDispatch();

    return (
        <>
            <IconButton
                aria-label='Settings'
                variant='link'
                fontSize={24}
                size={'sm'}
                icon={<MdSettings />}
                onClick={onOpen}
            />

            <Modal isOpen={isOpen} onClose={onClose}>
                <ModalOverlay />
                <ModalContent>
                    <ModalHeader>Settings</ModalHeader>
                    <ModalCloseButton />
                    <ModalBody>
                        <SDSwitch
                            label='Display in-progress images'
                            isChecked={shouldDisplayInProgress}
                            onChange={(e) =>
                                dispatch(
                                    setShouldDisplayInProgress(e.target.checked)
                                )
                            }
                        />

                        <SDSwitch
                            label='Fit to Width/Height'
                            isChecked={shouldFitToWidthHeight}
                            onChange={(e) =>
                                dispatch(
                                    setShouldFitToWidthHeight(e.target.checked)
                                )
                            }
                        />
                    </ModalBody>

                    <ModalFooter>
                        <SDButton label='Close' onClick={onClose} />
                    </ModalFooter>
                </ModalContent>
            </Modal>
        </>
    );
};

export default SettingsModalButton;
