import {
    Flex,
    IconButton,
    HStack,
    Spacer,
    ChakraProps,
    Box,
} from '@chakra-ui/react';

import { RootState } from '../../app/store';
import { useSocketIOEmitters } from '../../app/socket';
import { useAppDispatch, useAppSelector } from '../../app/hooks';

import { TiArrowBack } from 'react-icons/ti';
import { FaRandom } from 'react-icons/fa';

import {
    resetSDState,
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
    setShouldFitToWidthHeight,
    randomizeSeed,
    setSeamless,
} from '../sd/sdSlice';

import SDNumberInput from '../../components/SDNumberInput';
import SDSelect from '../../components/SDSelect';
import SDButton from '../../components/SDButton';

import {
    HEIGHTS,
    SAMPLERS,
    UPSCALING_LEVELS,
    WIDTHS,
} from '../../app/constants';
import SDSwitch from '../../components/SDSwitch';

const Settings = (props: ChakraProps) => {
    const {
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
        initialImagePath,
        shouldFitToWidthHeight,
        seamless,
    } = useAppSelector((state: RootState) => state.sd);

    const { isProcessing, isConnected, isGFPGANAvailable, isESRGANAvailable } =
        useAppSelector((state: RootState) => state.system);

    const dispatch = useAppDispatch();
    const { emitGenerateImage, emitCancel } = useSocketIOEmitters();

    return (
        <Flex direction={'column'} gap={2} {...props}>
            <HStack>
                <SDButton
                    label='Generate'
                    type='submit'
                    colorScheme='green'
                    isDisabled={!isConnected || isProcessing}
                    onClick={() => emitGenerateImage()}
                />
                <Spacer />
                <SDButton
                    label='Cancel'
                    colorScheme='red'
                    isDisabled={!isConnected || !isProcessing}
                    onClick={() => emitCancel()}
                />
                <Spacer />
                <SDButton
                    label='Reset'
                    colorScheme='blue'
                    onClick={() => dispatch(resetSDState())}
                />
            </HStack>

            <HStack>
                <SDNumberInput
                    label='Iterations'
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
            </HStack>
            <HStack>
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
            </HStack>
            <HStack>
                <Box flexGrow={3}>
                    <SDNumberInput
                        label='CFG Scale'
                        step={0.5}
                        onChange={(v) => dispatch(setCfgScale(Number(v)))}
                        value={cfgScale}
                    />
                </Box>
                <Box>
                    <SDSwitch
                        label='Seamless'
                        isChecked={seamless}
                        onChange={(e) =>
                            dispatch(setSeamless(e.target.checked))
                        }
                    />
                </Box>
            </HStack>
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
                    icon={<TiArrowBack />}
                    fontSize={20}
                    onClick={() => dispatch(resetSeed())}
                />
                <IconButton
                    aria-label='Randomize seed'
                    size={'sm'}
                    icon={<FaRandom />}
                    onClick={() => dispatch(randomizeSeed())}
                />
            </HStack>
            <SDSelect
                label='Sampler'
                value={sampler}
                onChange={(e) => dispatch(setSampler(e.target.value))}
                validValues={SAMPLERS}
            />
            <HStack>
                <Box flexGrow={3}>
                    <SDNumberInput
                        isDisabled={!initialImagePath}
                        label='i2i Strength'
                        step={0.01}
                        min={0}
                        max={1}
                        onChange={(v) =>
                            dispatch(setImg2imgStrength(Number(v)))
                        }
                        value={img2imgStrength}
                    />
                </Box>
                <Box>
                    <SDSwitch
                        isDisabled={!initialImagePath}
                        label='Fit'
                        isChecked={shouldFitToWidthHeight}
                        onChange={(e) =>
                            dispatch(
                                setShouldFitToWidthHeight(e.target.checked)
                            )
                        }
                    />
                </Box>
            </HStack>
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
        </Flex>
    );
};

export default Settings;
