import { Alert, Button, Flex, Grid, Text } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import { LaunchpadAddStyleReference } from './LaunchpadAddStyleReference';
import { LaunchpadContainer } from './LaunchpadContainer';
import { LaunchpadGenerateFromTextButton } from './LaunchpadGenerateFromTextButton';
import { LaunchpadStartingFrameButton } from './LaunchpadStartingFrameButton';
import { VideoModelPicker } from 'features/settingsAccordions/components/VideoSettingsAccordion/VideoModelPicker';

export const VideoLaunchpadPanel = memo(() => {
	const { t } = useTranslation();

	return (
		<LaunchpadContainer heading={t('ui.launchpad.videoTitle')}>
			<Grid gridTemplateColumns="1fr 1fr" gap={8}>
				<VideoModelPicker labelKey="common.selectYourModel" />
				<Flex flexDir="column" gap={2} justifyContent="center">
					<Text>
						{t('ui.launchpad.modelGuideText')}{' '}
						<Button
							as="a"
							variant="link"
							href="https://support.invoke.ai/support/solutions/articles/151000216086-model-guide"
							target="_blank"
							rel="noopener noreferrer"
							size="sm"
						>
							{t('ui.launchpad.modelGuideLink')}
						</Button>
					</Text>
				</Flex>
			</Grid>
			<LaunchpadGenerateFromTextButton />
			<LaunchpadStartingFrameButton />
		</LaunchpadContainer>
	);
});

VideoLaunchpadPanel.displayName = 'VideoLaunchpadPanel';