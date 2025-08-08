import { Flex, StandaloneAccordion } from '@invoke-ai/ui-library';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import { VideoFirstFrameImage } from './VideoFirstFrameImage';
import { VideoLastFrameImage } from './VideoLastFrameImage';


export const VideoSettingsAccordion = memo(() => {
    const { t } = useTranslation();
    const { isOpen: isOpenAccordion, onToggle: onToggleAccordion } = useStandaloneAccordionToggle({
        id: 'video-settings',
        defaultIsOpen: true,
    });


    return (
        <StandaloneAccordion
            label={t('upscaling.upscale')}
            badges={[]}
            isOpen={isOpenAccordion}
            onToggle={onToggleAccordion}
        >
            <Flex p={4} w="full" h="full" flexDir="column" data-testid="upscale-settings-accordion">
                <Flex  gap={4}>
                 
                        <VideoFirstFrameImage />
                        <VideoLastFrameImage />
                        

         

                </Flex>

            </Flex>
        </StandaloneAccordion>
    );
});

VideoSettingsAccordion.displayName = 'VideoSettingsAccordion';
