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
import { clearStorage } from 'app/store/enhancers/reduxRemember/driver';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import { selectShouldUseCPUNoise, shouldUseCpuNoiseChanged } from 'features/controlLayers/store/paramsSlice';
import { useRefreshAfterResetModal } from 'features/system/components/SettingsModal/RefreshAfterResetModal';
import { SettingsDeveloperLogIsEnabled } from 'features/system/components/SettingsModal/SettingsDeveloperLogIsEnabled';
import { SettingsDeveloperLogLevel } from 'features/system/components/SettingsModal/SettingsDeveloperLogLevel';
import { SettingsDeveloperLogNamespaces } from 'features/system/components/SettingsModal/SettingsDeveloperLogNamespaces';
import { useClearIntermediates } from 'features/system/components/SettingsModal/useClearIntermediates';
import { StickyScrollable } from 'features/system/components/StickyScrollable';
import {
  selectSystemShouldAntialiasProgressImage,
  selectSystemShouldConfirmOnDelete,
  selectSystemShouldConfirmOnNewSession,
  selectSystemShouldEnableHighlightFocusedRegions,
  selectSystemShouldEnableInformationalPopovers,
  selectSystemShouldEnableModelDescriptions,
  selectSystemShouldShowInvocationProgressDetail,
  selectSystemShouldUseNSFWChecker,
  selectSystemShouldUseWatermarker,
  setShouldConfirmOnDelete,
  setShouldEnableInformationalPopovers,
  setShouldEnableModelDescriptions,
  setShouldHighlightFocusedRegions,
  setShouldShowInvocationProgressDetail,
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

import { SettingsLanguageSelect } from './SettingsLanguageSelect';

const [useSettingsModal] = buildUseBoolean(false);

const SettingsModal = (props: { children: ReactElement }) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const {
    clearIntermediates,
    hasPendingItems,
    intermediatesCount,
    isLoading: isLoadingClearIntermediates,
    refetchIntermediatesCount,
  } = useClearIntermediates();

  const settingsModal = useSettingsModal();
  const refreshModal = useRefreshAfterResetModal();

  const shouldUseCpuNoise = useAppSelector(selectShouldUseCPUNoise);
  const shouldConfirmOnDelete = useAppSelector(selectSystemShouldConfirmOnDelete);
  const shouldShowProgressInViewer = useAppSelector(selectShouldShowProgressInViewer);
  const shouldAntialiasProgressImage = useAppSelector(selectSystemShouldAntialiasProgressImage);
  const shouldUseNSFWChecker = useAppSelector(selectSystemShouldUseNSFWChecker);
  const shouldUseWatermarker = useAppSelector(selectSystemShouldUseWatermarker);
  const shouldEnableInformationalPopovers = useAppSelector(selectSystemShouldEnableInformationalPopovers);
  const shouldEnableModelDescriptions = useAppSelector(selectSystemShouldEnableModelDescriptions);
  const shouldHighlightFocusedRegions = useAppSelector(selectSystemShouldEnableHighlightFocusedRegions);
  const shouldConfirmOnNewSession = useAppSelector(selectSystemShouldConfirmOnNewSession);
  const shouldShowInvocationProgressDetail = useAppSelector(selectSystemShouldShowInvocationProgressDetail);
  const onToggleConfirmOnNewSession = useCallback(() => {
    dispatch(shouldConfirmOnNewSessionToggled());
  }, [dispatch]);

  useEffect(() => {
    // Refetch intermediates count when modal is opened
    if (settingsModal.isTrue) {
      refetchIntermediatesCount();
    }
  }, [refetchIntermediatesCount, settingsModal.isTrue]);

  const handleClickResetWebUI = useCallback(() => {
    clearStorage();
    settingsModal.setFalse();
    refreshModal.setTrue();
  }, [settingsModal, refreshModal]);

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
  const handleChangeShouldEnableModelDescriptions = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setShouldEnableModelDescriptions(e.target.checked));
    },
    [dispatch]
  );
  const handleChangeShouldUseCpuNoise = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldUseCpuNoiseChanged(e.target.checked));
    },
    [dispatch]
  );

  const handleChangeShouldShowInvocationProgressDetail = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setShouldShowInvocationProgressDetail(e.target.checked));
    },
    [dispatch]
  );

  const handleChangeShouldHighlightFocusedRegions = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setShouldHighlightFocusedRegions(e.target.checked));
    },
    [dispatch]
  );

  return (
    <>
      {cloneElement(props.children, {
        onClick: settingsModal.setTrue,
      })}
      <Modal isOpen={settingsModal.isTrue} onClose={settingsModal.setFalse} size="2xl" isCentered useInert={false}>
        <ModalOverlay />
        <ModalContent maxH="80vh" h="68rem">
          <ModalHeader bg="none">{t('common.settingsLabel')}</ModalHeader>
          <ModalCloseButton tabIndex={1} />
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
                    <FormControl>
                      <FormLabel>{t('settings.enableNSFWChecker')}</FormLabel>
                      <Switch isChecked={shouldUseNSFWChecker} onChange={handleChangeShouldUseNSFWChecker} />
                    </FormControl>
                    <FormControl>
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
                      <FormLabel>{t('settings.showDetailedInvocationProgress')}</FormLabel>
                      <Switch
                        isChecked={shouldShowInvocationProgressDetail}
                        onChange={handleChangeShouldShowInvocationProgressDetail}
                      />
                    </FormControl>
                    <FormControl>
                      <InformationalPopover feature="noiseUseCPU" inPortal={false}>
                        <FormLabel>{t('parameters.useCpuNoise')}</FormLabel>
                      </InformationalPopover>
                      <Switch isChecked={shouldUseCpuNoise} onChange={handleChangeShouldUseCpuNoise} />
                    </FormControl>
                    <SettingsLanguageSelect />
                    <FormControl>
                      <FormLabel>{t('settings.enableInformationalPopovers')}</FormLabel>
                      <Switch
                        isChecked={shouldEnableInformationalPopovers}
                        onChange={handleChangeShouldEnableInformationalPopovers}
                      />
                    </FormControl>
                    <FormControl>
                      <FormLabel>{t('settings.enableModelDescriptions')}</FormLabel>
                      <Switch
                        isChecked={shouldEnableModelDescriptions}
                        onChange={handleChangeShouldEnableModelDescriptions}
                      />
                    </FormControl>
                    <FormControl>
                      <FormLabel>{t('settings.enableHighlightFocusedRegions')}</FormLabel>
                      <Switch
                        isChecked={shouldHighlightFocusedRegions}
                        onChange={handleChangeShouldHighlightFocusedRegions}
                      />
                    </FormControl>
                  </StickyScrollable>

                  <StickyScrollable title={t('settings.developer')}>
                    <SettingsDeveloperLogIsEnabled />
                    <SettingsDeveloperLogLevel />
                    <SettingsDeveloperLogNamespaces />
                  </StickyScrollable>

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

                  <StickyScrollable title={t('settings.resetWebUI')}>
                    <Button colorScheme="error" onClick={handleClickResetWebUI}>
                      {t('settings.resetWebUI')}
                    </Button>
                    <Text variant="subtext">{t('settings.resetWebUIDesc1')}</Text>
                    <Text variant="subtext">{t('settings.resetWebUIDesc2')}</Text>
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
