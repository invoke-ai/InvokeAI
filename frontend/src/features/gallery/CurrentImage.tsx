import {
    Center,
    Flex,
    IconButton,
    Image,
    Menu,
    MenuButton,
    MenuItem,
    MenuList,
    VStack,
} from '@chakra-ui/react';
import { BsThreeDots } from 'react-icons/bs';
import { FaUpload } from 'react-icons/fa';
import { MdDeleteForever } from 'react-icons/md';
import { useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import { useSocketIOEmitters } from '../../context/socket';

const height = 'calc(100vh - 176px)';

const CurrentImage = () => {
    const { currentImageUuid, images } = useAppSelector(
        (state: RootState) => state.gallery
    );
    const imageToDisplay = images.find(
        (image) => image.uuid === currentImageUuid
    );
    const { emitDeleteImage } = useSocketIOEmitters();
    return (
        <Center height={height}>
            {imageToDisplay && (
                <Flex gap={2}>
                    <Image
                        maxHeight={height}
                        src={imageToDisplay?.url}
                    />
                    <VStack>
                        <IconButton
                            aria-label='Delete image'
                            icon={<MdDeleteForever />}
                            fontSize={24}
                            onClick={() => emitDeleteImage(imageToDisplay?.uuid)}
                        />
                        <IconButton
                            aria-label='Load image settings'
                            icon={<FaUpload />}
                        />
                        <Menu>
                            <MenuButton as={IconButton} icon={<BsThreeDots />}>
                                Actions
                            </MenuButton>
                            <MenuList>
                                <MenuItem>Delete image</MenuItem>
                                <MenuItem>
                                    Use all parameters from this image
                                </MenuItem>
                                <MenuItem>Use Seed from this image</MenuItem>
                                <MenuItem>Use as initial image</MenuItem>
                                <MenuItem>Scale up using ESRGAN</MenuItem>
                                <MenuItem>Fix faces with GFPGAN</MenuItem>
                                <MenuItem>Generate Variations</MenuItem>
                            </MenuList>
                        </Menu>
                    </VStack>
                </Flex>
            )}
        </Center>
    );
};

export default CurrentImage;
