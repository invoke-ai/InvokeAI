import {
  Button,
  Flex,
  FormControl,
  FormControlGroup,
  FormLabel,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Switch,
  Text,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import { useClearStorage } from 'common/hooks/useClearStorage';
import { selectShouldUseCPUNoise, shouldUseCpuNoiseChanged } from 'features/controlLayers/store/paramsSlice';
import { useRefreshAfterResetModal } from 'features/system/components/SettingsModal/RefreshAfterResetModal';
import { SettingsDeveloperLogIsEnabled } from 'features/system/components/SettingsModal/SettingsDeveloperLogIsEnabled';
import { SettingsDeveloperLogLevel } from 'features/system/components/SettingsModal/SettingsDeveloperLogLevel';
import { SettingsDeveloperLogNamespaces } from 'features/system/components/SettingsModal/SettingsDeveloperLogNamespaces';
import { useClearIntermediates } from 'features/system/components/SettingsModal/useClearIntermediates';
import { StickyScrollable } from 'features/system/components/StickyScrollable';
import {
  logIsEnabledChanged,
  selectSystemShouldAntialiasProgressImage,
  selectSystemShouldConfirmOnDelete,
  selectSystemShouldConfirmOnNewSession,
  selectSystemShouldEnableInformationalPopovers,
  selectSystemShouldUseNSFWChecker,
  selectSystemShouldUseWatermarker,
  setShouldConfirmOnDelete,
  setShouldEnableInformationalPopovers,
  shouldAntialiasProgressImageChanged,
  shouldConfirmOnNewSessionToggled,
  shouldUseNSFWCheckerChanged,
  shouldUseWatermarkerChanged,
} from 'features/system/store/systemSlice';
import { selectShouldShowProgressInViewer } from 'features/ui/store/uiSelectors';
import { setShouldShowProgressInViewer } from 'features/ui/store/uiSlice';
import type { ChangeEvent, ReactElement } from 'react';
import { cloneElement, memo, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetAppConfigQuery } from 'services/api/endpoints/appInfo';

import { SettingsLanguageSelect } from './SettingsLanguageSelect';

type ConfigOptions = {
  shouldShowDeveloperSettings?: boolean;
  shouldShowResetWebUiText?: boolean;
  shouldShowClearIntermediates?: boolean;
  shouldShowLocalizationToggle?: boolean;
};

const defaultConfig: ConfigOptions = {
  shouldShowDeveloperSettings: true,
  shouldShowResetWebUiText: true,
  shouldShowClearIntermediates: true,
  shouldShowLocalizationToggle: true,
};

type SettingsModalProps = {
  /* The button to open the Settings Modal */
  children: ReactElement;
  config?: ConfigOptions;
};

const [useSettingsModal] = buildUseBoolean(false);

const SettingsModal = ({ config = defaultConfig, children }: SettingsModalProps) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  useEffect(() => {
    if (!config?.shouldShowDeveloperSettings) {
      dispatch(logIsEnabledChanged(false));
    }
  }, [dispatch, config?.shouldShowDeveloperSettings]);

  const { isNSFWCheckerAvailable, isWatermarkerAvailable } = useGetAppConfigQuery(undefined, {
    selectFromResult: ({ data }) => ({
      isNSFWCheckerAvailable: data?.nsfw_methods.includes('nsfw_checker') ?? false,
      isWatermarkerAvailable: data?.watermarking_methods.includes('invisible_watermark') ?? false,
    }),
  });

  const {
    clearIntermediates,
    hasPendingItems,
    intermediatesCount,
    isLoading: isLoadingClearIntermediates,
    refetchIntermediatesCount,
  } = useClearIntermediates(Boolean(config?.shouldShowClearIntermediates));
  const settingsModal = useSettingsModal();
  const refreshModal = useRefreshAfterResetModal();

  const shouldUseCpuNoise = useAppSelector(selectShouldUseCPUNoise);
  const shouldConfirmOnDelete = useAppSelector(selectSystemShouldConfirmOnDelete);
  const shouldShowProgressInViewer = useAppSelector(selectShouldShowProgressInViewer);
  const shouldAntialiasProgressImage = useAppSelector(selectSystemShouldAntialiasProgressImage);
  const shouldUseNSFWChecker = useAppSelector(selectSystemShouldUseNSFWChecker);
  const shouldUseWatermarker = useAppSelector(selectSystemShouldUseWatermarker);
  const shouldEnableInformationalPopovers = useAppSelector(selectSystemShouldEnableInformationalPopovers);
  const shouldConfirmOnNewSession = useAppSelector(selectSystemShouldConfirmOnNewSession);
  const onToggleConfirmOnNewSession = useCallback(() => {
    dispatch(shouldConfirmOnNewSessionToggled());
  }, [dispatch]);

  const clearStorage = useClearStorage();

  useEffect(() => {
    if (settingsModal.isTrue && Boolean(config?.shouldShowClearIntermediates)) {
      refetchIntermediatesCount();
    }
  }, [config?.shouldShowClearIntermediates, refetchIntermediatesCount, settingsModal.isTrue]);

  const handleClickResetWebUI = useCallback(() => {
    clearStorage();
    settingsModal.setFalse();
    refreshModal.setTrue();
  }, [clearStorage, settingsModal, refreshModal]);

  const handleChangeShouldConfirmOnDelete = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setShouldConfirmOnDelete(e.target.checked));
    },
    [dispatch]
  );
  const handleChangeShouldUseNSFWChecker = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldUseNSFWCheckerChanged(e.target.checked));
    },
    [dispatch]
  );
  const handleChangeShouldUseWatermarker = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldUseWatermarkerChanged(e.target.checked));
    },
    [dispatch]
  );
  const handleChangeShouldShowProgressInViewer = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setShouldShowProgressInViewer(e.target.checked));
    },
    [dispatch]
  );
  const handleChangeShouldAntialiasProgressImage = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldAntialiasProgressImageChanged(e.target.checked));
    },
    [dispatch]
  );
  const handleChangeShouldEnableInformationalPopovers = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setShouldEnableInformationalPopovers(e.target.checked));
    },
    [dispatch]
  );
  const handleChangeShouldUseCpuNoise = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldUseCpuNoiseChanged(e.target.checked));
    },
    [dispatch]
  );

  return (
    <>
      {cloneElement(children, {
        onClick: settingsModal.setTrue,
      })}
      <Modal isOpen={settingsModal.isTrue} onClose={settingsModal.setFalse} size="2xl" isCentered useInert={false}>
        <ModalOverlay />
        <ModalContent maxH="80vh" h="68rem">
          <ModalHeader bg="none">{t('common.settingsLabel')}</ModalHeader>
          <ModalCloseButton />
          <ModalBody display="flex" flexDir="column" gap={4}>
            <ScrollableContent>
              <Flex flexDir="column" gap={4}>
                <FormControlGroup formLabelProps={{ flexGrow: 1 }}>
                  <StickyScrollable title={t('settings.general')}>
                    <FormControl>
                      <FormLabel>{t('settings.confirmOnDelete')}</FormLabel>
                      <Switch isChecked={shouldConfirmOnDelete} onChange={handleChangeShouldConfirmOnDelete} />
                    </FormControl>
                    <FormControl>
                      <FormLabel>{t('settings.confirmOnNewSession')}</FormLabel>
                      <Switch isChecked={shouldConfirmOnNewSession} onChange={onToggleConfirmOnNewSession} />
                    </FormControl>
                  </StickyScrollable>

                  <StickyScrollable title={t('settings.generation')}>
                    <FormControl isDisabled={!isNSFWCheckerAvailable}>
                      <FormLabel>{t('settings.enableNSFWChecker')}</FormLabel>
                      <Switch isChecked={shouldUseNSFWChecker} onChange={handleChangeShouldUseNSFWChecker} />
                    </FormControl>
                    <FormControl isDisabled={!isWatermarkerAvailable}>
                      <FormLabel>{t('settings.enableInvisibleWatermark')}</FormLabel>
                      <Switch isChecked={shouldUseWatermarker} onChange={handleChangeShouldUseWatermarker} />
                    </FormControl>
                  </StickyScrollable>

                  <StickyScrollable title={t('settings.ui')}>
                    <FormControl>
                      <FormLabel>{t('settings.showProgressInViewer')}</FormLabel>
                      <Switch
                        isChecked={shouldShowProgressInViewer}
                        onChange={handleChangeShouldShowProgressInViewer}
                      />
                    </FormControl>
                    <FormControl>
                      <FormLabel>{t('settings.antialiasProgressImages')}</FormLabel>
                      <Switch
                        isChecked={shouldAntialiasProgressImage}
                        onChange={handleChangeShouldAntialiasProgressImage}
                      />
                    </FormControl>
                    <FormControl>
                      <InformationalPopover feature="noiseUseCPU" inPortal={false}>
                        <FormLabel>{t('parameters.useCpuNoise')}</FormLabel>
                      </InformationalPopover>
                      <Switch isChecked={shouldUseCpuNoise} onChange={handleChangeShouldUseCpuNoise} />
                    </FormControl>
                    {Boolean(config?.shouldShowLocalizationToggle) && <SettingsLanguageSelect />}
                    <FormControl>
                      <FormLabel>{t('settings.enableInformationalPopovers')}</FormLabel>
                      <Switch
                        isChecked={shouldEnableInformationalPopovers}
                        onChange={handleChangeShouldEnableInformationalPopovers}
                      />
                    </FormControl>
                  </StickyScrollable>

                  {Boolean(config?.shouldShowDeveloperSettings) && (
                    <StickyScrollable title={t('settings.developer')}>
                      <SettingsDeveloperLogIsEnabled />
                      <SettingsDeveloperLogLevel />
                      <SettingsDeveloperLogNamespaces />
                    </StickyScrollable>
                  )}

                  {Boolean(config?.shouldShowClearIntermediates) && (
                    <StickyScrollable title={t('settings.clearIntermediates')}>
                      <Button
                        tooltip={hasPendingItems ? t('settings.clearIntermediatesDisabled') : undefined}
                        colorScheme="warning"
                        onClick={clearIntermediates}
                        isLoading={isLoadingClearIntermediates}
                        isDisabled={!intermediatesCount || hasPendingItems}
                      >
                        {t('settings.clearIntermediatesWithCount', {
                          count: intermediatesCount ?? 0,
                        })}
                      </Button>
                      <Text fontWeight="bold">{t('settings.clearIntermediatesDesc1')}</Text>
                      <Text variant="subtext">{t('settings.clearIntermediatesDesc2')}</Text>
                      <Text variant="subtext">{t('settings.clearIntermediatesDesc3')}</Text>
                    </StickyScrollable>
                  )}

                  <StickyScrollable title={t('settings.resetWebUI')}>
                    <Button colorScheme="error" onClick={handleClickResetWebUI}>
                      {t('settings.resetWebUI')}
                    </Button>
                    {Boolean(config?.shouldShowResetWebUiText) && (
                      <>
                        <Text variant="subtext">{t('settings.resetWebUIDesc1')}</Text>
                        <Text variant="subtext">{t('settings.resetWebUIDesc2')}</Text>
                      </>
                    )}
                  </StickyScrollable>
                </FormControlGroup>
              </Flex>
            </ScrollableContent>
          </ModalBody>

          <ModalFooter />
        </ModalContent>
      </Modal>
    </>
  );
};

export default memo(SettingsModal);
