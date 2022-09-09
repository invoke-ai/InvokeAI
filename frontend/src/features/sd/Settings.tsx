import {
    Flex,
    IconButton,
    HStack,
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
    setIterations,
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
import useCheckParameters from '../system/useCheckParameters';

const Settings = () => {
    const {
        iterations,
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
        shouldGenerateVariations,
    } = useAppSelector((state: RootState) => state.sd);

    const { isProcessing, isConnected, isGFPGANAvailable, isESRGANAvailable } =
        useAppSelector((state: RootState) => state.system);

    const dispatch = useAppDispatch();

    const { emitGenerateImage, emitCancel } = useSocketIOEmitters();

    const areParametersValid = useCheckParameters();

    return (
        <Flex direction={'column'} gap={2}>
            <HStack justifyContent={'stretch'}>
                {isProcessing ? (
                    <SDButton
                        label='Cancel'
                        colorScheme='red'
                        isDisabled={!isConnected || !isProcessing}
                        onClick={() => emitCancel()}
                    />
                ) : (
                    <SDButton
                        label='Generate'
                        type='submit'
                        colorScheme='green'
                        isDisabled={!areParametersValid}
                        onClick={() => emitGenerateImage()}
                    />
                )}
                {/*<Spacer />*/}
                <SDButton
                    label='Reset all image parameters'
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
                    onChange={(v) => dispatch(setIterations(Number(v)))}
                    value={iterations}
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
                        label='CFG scale'
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
                    isInvalid={seed < 0 && shouldGenerateVariations}
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
