import { Flex, StandaloneAccordion } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { aspectRatioIdChanged, aspectRatioLockToggled,heightChanged, widthChanged } from 'features/controlLayers/store/paramsSlice';
import { RUNWAY_ASPECT_RATIOS } from 'features/controlLayers/store/types';
import { Dimensions } from 'features/parameters/components/Dimensions/Dimensions';
import { ParamSeed } from 'features/parameters/components/Seed/ParamSeed';
import { ParamDuration } from 'features/parameters/components/Video/ParamDuration';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { memo, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

import { StartingFrameImage } from './StartingFrameImage';


export const VideoSettingsAccordion = memo(() => {
    const { t } = useTranslation();
    const { isOpen: isOpenAccordion, onToggle: onToggleAccordion } = useStandaloneAccordionToggle({
        id: 'video-settings',
        defaultIsOpen: true,
    });
    const dispatch = useAppDispatch();

    useEffect(() => { // hack to get the default aspect ratio for runway models outside paramsSlice
        const { width, height } = RUNWAY_ASPECT_RATIOS['16:9'];
        dispatch(widthChanged({ width, updateAspectRatio: true, clamp: true }));
        dispatch(heightChanged({ height, updateAspectRatio: true, clamp: true }));
        dispatch(aspectRatioIdChanged({ id: '16:9' }));
        dispatch(aspectRatioLockToggled());
    }, [dispatch]);


    return (
        <StandaloneAccordion
            label="Video"
            badges={[]}
            isOpen={isOpenAccordion}
            onToggle={onToggleAccordion}
        >
            <Flex p={4} w="full" h="full" flexDir="column" data-testid="upscale-settings-accordion">
                <Flex gap={4} flexDirection="column" width="full">
                <Flex gap={4}>
                    <StartingFrameImage />
                    <Flex gap={4} flexDirection="column" width="full">
                        <ParamDuration />
                    </Flex>
                </Flex>
                <Dimensions />
                <ParamSeed />
                </Flex>
            </Flex>
        </StandaloneAccordion>
    );
});

VideoSettingsAccordion.displayName = 'VideoSettingsAccordion';
