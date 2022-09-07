import {
    Flex,
    IconButton,
    Modal,
    ModalBody,
    ModalCloseButton,
    ModalContent,
    ModalFooter,
    ModalHeader,
    ModalOverlay,
    Popover,
    PopoverArrow,
    PopoverBody,
    PopoverCloseButton,
    PopoverContent,
    PopoverHeader,
    PopoverTrigger,
    Text,
    useDisclosure,
    useToast,
} from '@chakra-ui/react';
import { MdSettings } from 'react-icons/md';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { setShouldDisplayInProgress } from './systemSlice';
import { RootState } from '../../app/store';
import SDButton from '../../components/SDButton';
import SDSwitch from '../../components/SDSwitch';
import { persistor } from '../../main';
import { useState } from 'react';

const SettingsModalButton = () => {
    const { isOpen, onOpen, onClose } = useDisclosure();
    const { shouldDisplayInProgress } = useAppSelector(
        (state: RootState) => state.system
    );
    const [isResetting, setIsResetting] = useState<boolean>(false);
    const toast = useToast();

    const dispatch = useAppDispatch();

    const handleClickResetWebUI = () => {
        setIsResetting(true);
        persistor.purge().then(() => {
            toast({
                title: 'Web UI reset',
                status: 'success',
                isClosable: true,
            });
            setIsResetting(false);
        });
    };

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
                        <Flex gap={5} direction='column'>
                            <SDSwitch
                                label='Display in-progress images'
                                isChecked={shouldDisplayInProgress}
                                onChange={(e) =>
                                    dispatch(
                                        setShouldDisplayInProgress(
                                            e.target.checked
                                        )
                                    )
                                }
                            />
                            <SDButton
                                label='Reset Web UI'
                                colorScheme='orange'
                                onClick={handleClickResetWebUI}
                                isLoading={isResetting}
                            />
                            <Text>
                                Resetting the web UI only resets the browser's
                                local cache of your images and remembered
                                settings. It does not delete any images from
                                disk. After resetting, refresh your browser to
                                re-load all images into the web UI.
                            </Text>
                        </Flex>
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
