import {
    IconButton,
    useColorModeValue,
    UnorderedList,
    ListItem,
    Flex,
} from '@chakra-ui/react';
import { useAppSelector } from '../app/hooks';
import { IoIosArrowDown, IoIosArrowUp } from 'react-icons/io';
import { useState } from 'react';
import { RootState } from '../app/store';

const LogViewer = () => {
    const [shouldShowLogViewer, setShouldShowLogViewer] =
        useState<boolean>(false);
    const bg = useColorModeValue('blue.50', 'blue.900');
    const borderColor = useColorModeValue('blue.500', 'blue.500');

    const { log } = useAppSelector((state: RootState) => state.system);

    return (
        <>
            {shouldShowLogViewer && (
                <Flex
                    position={'fixed'}
                    left={0}
                    bottom={0}
                    height='200px'
                    width='100vw'
                    overflow='auto'
                    direction='column-reverse'
                    fontFamily='monospace'
                    fontSize='sm'
                    pl={0}
                    pr={2}
                    pb={2}
                    background={bg}
                    borderTopWidth='4px'
                    borderColor={borderColor}
                >
                    <UnorderedList listStyleType='none'>
                        {log.map((line) => (
                            <ListItem key={line}>{line}</ListItem>
                        ))}
                    </UnorderedList>
                </Flex>
            )}
            <IconButton
                size='sm'
                position={'fixed'}
                left={2}
                bottom={2}
                aria-label='Toggle Log Viewer'
                icon={
                    shouldShowLogViewer ? <IoIosArrowDown /> : <IoIosArrowUp />
                }
                onClick={() => setShouldShowLogViewer(!shouldShowLogViewer)}
            />
        </>
    );
};

export default LogViewer;
