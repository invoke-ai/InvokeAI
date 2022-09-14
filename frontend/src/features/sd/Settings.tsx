import { Flex, IconButton, HStack, Box } from '@chakra-ui/react';

import { RootState } from '../../app/store';
import { useAppDispatch, useAppSelector } from '../../app/hooks';

import { FaRandom } from 'react-icons/fa';

import {
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
    UpscalingLevel,
    setShouldRandomizeSeed,
    setShouldRunGFPGAN,
    setShouldRunESRGAN,
    SDState,
} from '../sd/sdSlice';

import SDNumberInput from '../../components/SDNumberInput';
import SDSelect from '../../components/SDSelect';

import {
    HEIGHTS,
    NUMPY_RAND_MAX,
    NUMPY_RAND_MIN,
    SAMPLERS,
    UPSCALING_LEVELS,
    WIDTHS,
} from '../../app/constants';
import SDSwitch from '../../components/SDSwitch';
import ProcessButtons from './ProcessButtons';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { SystemState } from '../system/systemSlice';

const sdSelector = createSelector(
    (state: RootState) => state.sd,
    (sd: SDState) => {
        return {
            iterations: sd.iterations,
            steps: sd.steps,
            cfgScale: sd.cfgScale,
            height: sd.height,
            width: sd.width,
            sampler: sd.sampler,
            seed: sd.seed,
            img2imgStrength: sd.img2imgStrength,
            gfpganStrength: sd.gfpganStrength,
            upscalingLevel: sd.upscalingLevel,
            upscalingStrength: sd.upscalingStrength,
            initialImagePath: sd.initialImagePath,
            shouldFitToWidthHeight: sd.shouldFitToWidthHeight,
            seamless: sd.seamless,
            shouldGenerateVariations: sd.shouldGenerateVariations,
            shouldRandomizeSeed: sd.shouldRandomizeSeed,
            shouldRunESRGAN: sd.shouldRunESRGAN,
            shouldRunGFPGAN: sd.shouldRunGFPGAN,
        };
    },
    {
        memoizeOptions: {
            resultEqualityCheck: isEqual,
        },
    }
);

const systemSelector = createSelector(
    (state: RootState) => state.system,
    (system: SystemState) => {
        return {
            isGFPGANAvailable: system.isGFPGANAvailable,
            isESRGANAvailable: system.isESRGANAvailable,
        };
    },
    {
        memoizeOptions: {
            resultEqualityCheck: isEqual,
        },
    }
);

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
        shouldRandomizeSeed,
        shouldRunESRGAN,
        shouldRunGFPGAN,
    } = useAppSelector(sdSelector);

    const { isGFPGANAvailable, isESRGANAvailable } =
        useAppSelector(systemSelector);

    const dispatch = useAppDispatch();

    return (
        <Flex direction={'column'} gap={2}>
            <ProcessButtons />
            <Flex>
                <SDSwitch
                    isDisabled={!isESRGANAvailable}
                    label='Run ESRGAN'
                    isChecked={shouldRunESRGAN}
                    onChange={(e) =>
                        dispatch(setShouldRunESRGAN(e.target.checked))
                    }
                />
                <SDSwitch
                    isDisabled={!isGFPGANAvailable}
                    label='Run GFPGAN'
                    isChecked={shouldRunGFPGAN}
                    onChange={(e) =>
                        dispatch(setShouldRunGFPGAN(e.target.checked))
                    }
                />
            </Flex>
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
                <Box>
                    <SDSwitch
                        label='Random'
                        isChecked={shouldRandomizeSeed}
                        onChange={(e) =>
                            dispatch(setShouldRandomizeSeed(e.target.checked))
                        }
                    />
                </Box>
                <Box flexGrow={3}>
                    <SDNumberInput
                        label='Seed'
                        step={1}
                        precision={0}
                        min={NUMPY_RAND_MIN}
                        max={NUMPY_RAND_MAX}
                        isDisabled={shouldRandomizeSeed}
                        isInvalid={seed < 0 && shouldGenerateVariations}
                        onChange={(v) => dispatch(setSeed(Number(v)))}
                        value={seed}
                    />
                </Box>
                <IconButton
                    aria-label='Randomize '
                    size={'sm'}
                    icon={<FaRandom />}
                    isDisabled={shouldRandomizeSeed}
                    onClick={() => dispatch(randomizeSeed())}
                />
            </HStack>
            <SDSelect
                label='Sampler'
                value={sampler}
                onChange={(e) => dispatch(setSampler(e.target.value))}
                validValues={SAMPLERS}
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
                    dispatch(
                        setUpscalingLevel(
                            Number(e.target.value) as UpscalingLevel
                        )
                    )
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
        </Flex>
    );
};

export default Settings;
