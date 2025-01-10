import type { ButtonProps } from '@invoke-ai/ui-library';
import { Alert, AlertDescription, AlertIcon, Button, Divider, Flex, Link, Spinner, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { InvokeLogoIcon } from 'common/components/InvokeLogoIcon';
import { LOADING_SYMBOL, useHasImages } from 'features/gallery/hooks/useHasImages';
import { $installModelsTab } from 'features/modelManagerV2/subpanels/InstallModels';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { selectIsLocal } from 'features/system/store/configSlice';
import { setActiveTab } from 'features/ui/store/uiSlice';
import type { PropsWithChildren } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { PiArrowSquareOutBold, PiImageBold } from 'react-icons/pi';
import { useMainModels } from 'services/api/hooks/modelsByType';

export const NoContentForViewer = memo(() => {
  const hasImages = useHasImages();
  const [mainModels, { data }] = useMainModels();
  const isLocal = useAppSelector(selectIsLocal);
  const isEnabled = useFeatureStatus('starterModels');
  const { t } = useTranslation();

  const showStarterBundles = useMemo(() => {
    return isEnabled && data && mainModels.length === 0;
  }, [mainModels.length, data, isEnabled]);

  if (hasImages === LOADING_SYMBOL) {
    // Blank bg w/ a spinner. The new user experience components below have an invoke logo, but it's not centered.
    // If we show the logo while loading, there is an awkward layout shift where the invoke logo moves a bit. Less
    // jarring to show a blank bg with a spinner - it will only be shown for a moment as we do the initial images
    // fetching.
    return <LoadingSpinner />;
  }

  if (hasImages) {
    return <IAINoContentFallback icon={PiImageBold} label={t('gallery.noImageSelected')} />;
  }

  return (
    <Flex flexDir="column" gap={8} alignItems="center" textAlign="center" maxW="600px">
      <InvokeLogoIcon w={32} h={32} />
      <Flex flexDir="column" gap={4} alignItems="center" textAlign="center">
        {isLocal ? <GetStartedLocal /> : <GetStartedCommercial />}
        {showStarterBundles && <StarterBundlesCallout />}
        <Divider />
        <GettingStartedVideosCallout />
        {isLocal && <LowVRAMAlert />}
      </Flex>
    </Flex>
  );
});

NoContentForViewer.displayName = 'NoContentForViewer';

const LoadingSpinner = () => {
  return (
    <Flex position="relative" width="full" height="full" alignItems="center" justifyContent="center">
      <Spinner label="Loading" color="grey" position="absolute" size="sm" width={8} height={8} right={4} bottom={4} />
    </Flex>
  );
};

export const ExternalLink = (props: ButtonProps & { href: string }) => {
  return (
    <Button
      as={Link}
      variant="unstyled"
      isExternal
      display="inline-flex"
      alignItems="center"
      rightIcon={<PiArrowSquareOutBold />}
      color="base.50"
      mt={-1}
      {...props}
    />
  );
};

const InlineButton = (props: PropsWithChildren<{ onClick: () => void }>) => {
  return (
    <Button variant="link" size="md" onClick={props.onClick} color="base.50">
      {props.children}
    </Button>
  );
};

const StrongComponent = <Text as="span" color="base.50" fontSize="md" />;

const GetStartedLocal = () => {
  return (
    <Text fontSize="md" color="base.200">
      <Trans i18nKey="newUserExperience.toGetStartedLocal" components={{ StrongComponent }} />
    </Text>
  );
};

const GetStartedCommercial = () => {
  return (
    <Text fontSize="md" color="base.200">
      <Trans i18nKey="newUserExperience.toGetStarted" components={{ StrongComponent }} />
    </Text>
  );
};

const GettingStartedVideosCallout = () => {
  return (
    <Text fontSize="md" color="base.200">
      <Trans
        i18nKey="newUserExperience.gettingStartedSeries"
        components={{
          LinkComponent: (
            <ExternalLink href="https://www.youtube.com/playlist?list=PLvWK1Kc8iXGrQy8r9TYg6QdUuJ5MMx-ZO" />
          ),
        }}
      />
    </Text>
  );
};

const StarterBundlesCallout = () => {
  const dispatch = useAppDispatch();

  const handleClickDownloadStarterModels = useCallback(() => {
    dispatch(setActiveTab('models'));
    $installModelsTab.set(3);
  }, [dispatch]);

  const handleClickImportModels = useCallback(() => {
    dispatch(setActiveTab('models'));
    $installModelsTab.set(0);
  }, [dispatch]);

  return (
    <Text fontSize="md" color="base.200">
      <Trans
        i18nKey="newUserExperience.noModelsInstalled"
        components={{
          DownloadStarterModelsButton: <InlineButton onClick={handleClickDownloadStarterModels} />,
          ImportModelsButton: <InlineButton onClick={handleClickImportModels} />,
        }}
      />
    </Text>
  );
};

const LowVRAMAlert = () => {
  return (
    <Alert status="warning" borderRadius="base" fontSize="md" shadow="md" w="fit-content">
      <AlertIcon />
      <AlertDescription>
        <Trans
          i18nKey="newUserExperience.lowVRAMMode"
          components={{
            LinkComponent: <ExternalLink href="https://invoke-ai.github.io/InvokeAI/features/low-vram/" />,
          }}
        />
      </AlertDescription>
    </Alert>
  );
};
