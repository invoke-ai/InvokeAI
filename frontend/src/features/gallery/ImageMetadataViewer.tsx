import {
    Center,
    Flex,
    IconButton,
    List,
    ListItem,
    Text,
} from '@chakra-ui/react';
import { FaPlus } from 'react-icons/fa';
import { PARAMETERS } from '../../app/constants';
import { useAppDispatch } from '../../app/hooks';
import { setParameter } from '../sd/sdSlice';
import { SDImage, SDMetadata } from './gallerySlice';

type Props = {
    image: SDImage;
};

const ImageMetadataViewer = ({ image }: Props) => {
    const dispatch = useAppDispatch();

    const keys = Object.keys(PARAMETERS);

    const metadata: Array<{
        label: string;
        key: string;
        value: string | number | boolean;
    }> = [];

    keys.forEach((key) => {
        const value = image.metadata[key as keyof SDMetadata];
        if (value !== undefined) {
            metadata.push({ label: PARAMETERS[key], key, value });
        }
    });

    return metadata.length ? (
        <List>
            {metadata.map((parameter) => {
                const { label, key, value } = parameter;
                return value ? (
                    <ListItem pb={1}>
                        <Flex gap={2}>
                            <IconButton
                                aria-label='Use this parameter'
                                icon={<FaPlus />}
                                size={'xs'}
                                onClick={() =>
                                    dispatch(
                                        setParameter({
                                            key,
                                            value,
                                        })
                                    )
                                }
                            />
                            <Text fontWeight={'semibold'}>{label}:</Text>
                            <Text>{value.toString()}</Text>
                        </Flex>
                    </ListItem>
                ) : null;
            })}
        </List>
    ) : (
        <Center width={'100%'}>
            <Text fontSize={'lg'} fontWeight='semibold'>
                No metadata available
            </Text>
        </Center>
    );
};

export default ImageMetadataViewer;
