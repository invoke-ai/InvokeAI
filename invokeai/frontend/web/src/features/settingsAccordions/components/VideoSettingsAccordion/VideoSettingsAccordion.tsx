import { Flex, StandaloneAccordion } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { Dimensions } from 'features/parameters/components/Dimensions/Dimensions';
import { ParamSeed } from 'features/parameters/components/Seed/ParamSeed';
import { ParamDuration } from 'features/parameters/components/Video/ParamDuration';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { memo, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

import { StartingFrameImage } from './StartingFrameImage';
import { VideoModelPicker } from './VideoModelPicker';
import { ParamResolution } from 'features/parameters/components/Video/ParamResolution';


export const VideoSettingsAccordion = memo(() => {
    const { t } = useTranslation();
    const { isOpen: isOpenAccordion, onToggle: onToggleAccordion } = useStandaloneAccordionToggle({
        id: 'video-settings',
        defaultIsOpen: true,
    });
    const dispatch = useAppDispatch();

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
                        <VideoModelPicker />
                        <ParamDuration />
                        <ParamResolution />
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
