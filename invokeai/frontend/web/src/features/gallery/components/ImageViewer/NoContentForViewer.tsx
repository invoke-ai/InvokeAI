import { Button, Divider, Flex, Spinner, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { InvokeLogoIcon } from 'common/components/InvokeLogoIcon';
import { LOADING_SYMBOL, useHasImages } from 'features/gallery/hooks/useHasImages';
import { $installModelsTab } from 'features/modelManagerV2/subpanels/InstallModels';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { selectIsLocal } from 'features/system/store/configSlice';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useCallback, useMemo } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { PiImageBold } from 'react-icons/pi';
import { useMainModels } from 'services/api/hooks/modelsByType';

export const NoContentForViewer = () => {
  const hasImages = useHasImages();
  const [mainModels, { data }] = useMainModels();
  const isLocal = useAppSelector(selectIsLocal);
  const isEnabled = useFeatureStatus('starterModels');
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const handleClickDownloadStarterModels = useCallback(() => {
    dispatch(setActiveTab('models'));
    $installModelsTab.set(3);
  }, [dispatch]);

  const handleClickImportModels = useCallback(() => {
    dispatch(setActiveTab('models'));
    $installModelsTab.set(0);
  }, [dispatch]);

  const showStarterBundles = useMemo(() => {
    return isEnabled && data && mainModels.length === 0;
  }, [mainModels.length, data, isEnabled]);

  if (hasImages === LOADING_SYMBOL) {
    return (
      // Blank bg w/ a spinner. The new user experience components below have an invoke logo, but it's not centered.
      // If we show the logo while loading, there is an awkward layout shift where the invoke logo moves a bit. Less
      // jarring to show a blank bg with a spinner - it will only be shown for a moment as we do the initial images
      // fetching.
      <Flex position="relative" width="full" height="full" alignItems="center" justifyContent="center">
        <Spinner label="Loading" color="grey" position="absolute" size="sm" width={8} height={8} right={4} bottom={4} />
      </Flex>
    );
  }

  if (hasImages) {
    return <IAINoContentFallback icon={PiImageBold} label={t('gallery.noImageSelected')} />;
  }

  return (
    <Flex flexDir="column" gap={4} alignItems="center" textAlign="center" maxW="600px">
      <InvokeLogoIcon w={40} h={40} />
      <Flex flexDir="column" gap={8} alignItems="center" textAlign="center">
        <Text fontSize="md" color="base.200" pt={16}>
          {isLocal ? (
            <Trans
              i18nKey="newUserExperience.toGetStartedLocal"
              components={{
                StrongComponent: <Text as="span" color="white" fontSize="md" fontWeight="semibold" />,
              }}
            />
          ) : (
            <Trans
              i18nKey="newUserExperience.toGetStarted"
              components={{
                StrongComponent: <Text as="span" color="white" fontSize="md" fontWeight="semibold" />,
              }}
            />
          )}
        </Text>

        {showStarterBundles && (
          <Flex flexDir="column" gap={2} alignItems="center">
            <Text fontSize="md" color="base.200">
              {t('newUserExperience.noModelsInstalled')}
            </Text>
            <Flex gap={3} alignItems="center">
              <Button size="sm" onClick={handleClickDownloadStarterModels}>
                {t('newUserExperience.downloadStarterModels')}
              </Button>
              <Text fontSize="sm" color="base.200">
                {t('common.or')}
              </Text>
              <Button size="sm" onClick={handleClickImportModels}>
                {t('newUserExperience.importModels')}
              </Button>
            </Flex>
          </Flex>
        )}

        <Divider />

        <Text fontSize="md" color="base.200">
          <Trans
            i18nKey="newUserExperience.gettingStartedSeries"
            components={{
              LinkComponent: (
                <Text
                  as="a"
                  color="white"
                  fontSize="md"
                  fontWeight="semibold"
                  href="https://www.youtube.com/playlist?list=PLvWK1Kc8iXGrQy8r9TYg6QdUuJ5MMx-ZO"
                  target="_blank"
                />
              ),
            }}
          />
        </Text>
      </Flex>
    </Flex>
  );
};
