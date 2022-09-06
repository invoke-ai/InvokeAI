import { Flex, IconButton, HStack, Box, Spacer } from '@chakra-ui/react';

import { RootState } from '../../app/store';

import { useAppDispatch, useAppSelector } from '../../app/hooks';

import { BsArrowCounterclockwise } from 'react-icons/bs';

import {
    resetForm,
    resetSeed,
    setCfgScale,
    setGfpganStrength,
    setHeight,
    setImagesToGenerate,
    setImg2imgStrength,
    setSampler,
    setSeed,
    setSteps,
    setUpscalingLevel,
    setUpscalingStrength,
    setWidth,
} from '../../app/sdSlice';

import SDNumberInput from '../../components/SDNumberInput';
import SDSelect from '../../components/SDSelect';
import SDButton from '../../components/SDButton';
import SDFileUpload from '../../components/SDFileUpload';

import {
    HEIGHTS,
    SAMPLERS,
    UPSCALING_LEVELS,
    WIDTHS,
} from '../../app/constants';
import { useSocketIOEmitters } from '../../context/socket';

const DreamMenu = () => {
    const {
        isProcessing,
        isConnected,
        imagesToGenerate,
        steps,
        cfgScale,
        height,
        width,
        sampler,
        seed,
        img2imgStrength,
        gfpganStrength,
        upscalingLevel,
        upscalingStrength,
        isGFPGANAvailable,
        isESRGANAvailable,
    } = useAppSelector((state: RootState) => state.sd);

    const dispatch = useAppDispatch();
    const { generateImage, cancel } = useSocketIOEmitters();

    return (
        <Box>
            <Flex direction={'column'} gap={2}>
                <Flex>
                    <SDButton
                        label='Generate'
                        type='submit'
                        colorScheme='green'
                        isDisabled={!isConnected || isProcessing}
                        onClick={() => generateImage()}
                    />
                    <Spacer />
                    <SDButton
                        label='Cancel'
                        colorScheme='red'
                        isDisabled={!isConnected && !isProcessing}
                        onClick={() => cancel()}
                    />
                    <Spacer />
                    <SDButton
                        label='Reset'
                        colorScheme='blue'
                        onClick={() => dispatch(resetForm())}
                    />
                </Flex>

                <SDNumberInput
                    label='Image Count'
                    step={1}
                    min={1}
                    precision={0}
                    onChange={(v) => dispatch(setImagesToGenerate(Number(v)))}
                    value={imagesToGenerate}
                />

                <SDNumberInput
                    label='Steps'
                    min={1}
                    step={1}
                    precision={0}
                    onChange={(v) => dispatch(setSteps(Number(v)))}
                    value={steps}
                />

                <SDNumberInput
                    label='CFG Scale'
                    step={0.5}
                    onChange={(v) => dispatch(setCfgScale(Number(v)))}
                    value={cfgScale}
                />

                <HStack>
                    <SDNumberInput
                        label='Seed'
                        step={1}
                        precision={0}
                        onChange={(v) => dispatch(setSeed(Number(v)))}
                        value={seed}
                    />

                    <IconButton
                        aria-label='Reset seed to default'
                        size={'sm'}
                        icon={<BsArrowCounterclockwise />}
                        onClick={() => dispatch(resetSeed())}
                    />
                </HStack>

                <SDSelect
                    label='Sampler'
                    value={sampler}
                    onChange={(e) => dispatch(setSampler(e.target.value))}
                    validValues={SAMPLERS}
                />

                <SDSelect
                    label='Width'
                    value={width}
                    onChange={(e) => dispatch(setWidth(Number(e.target.value)))}
                    validValues={WIDTHS}
                />

                <SDSelect
                    label='Height'
                    value={height}
                    onChange={(e) =>
                        dispatch(setHeight(Number(e.target.value)))
                    }
                    validValues={HEIGHTS}
                />

                <SDNumberInput
                    label='img2img Strength'
                    step={0.01}
                    min={0}
                    max={1}
                    onChange={(v) => dispatch(setImg2imgStrength(Number(v)))}
                    value={img2imgStrength}
                />

                <SDNumberInput
                    isDisabled={!isGFPGANAvailable}
                    label='GFPGAN Strength'
                    step={0.05}
                    min={0}
                    max={1}
                    onChange={(v) => dispatch(setGfpganStrength(Number(v)))}
                    value={gfpganStrength}
                />

                <SDSelect
                    isDisabled={!isESRGANAvailable}
                    label='Upscaling Level'
                    value={upscalingLevel}
                    onChange={(e) =>
                        dispatch(setUpscalingLevel(Number(e.target.value)))
                    }
                    validValues={UPSCALING_LEVELS}
                />

                <SDNumberInput
                    isDisabled={!isESRGANAvailable}
                    label='Upscaling Strength'
                    step={0.05}
                    min={0}
                    max={1}
                    onChange={(v) => dispatch(setUpscalingStrength(Number(v)))}
                    value={upscalingStrength}
                />

                <SDFileUpload />
            </Flex>
        </Box>
    );
};

export default DreamMenu;
