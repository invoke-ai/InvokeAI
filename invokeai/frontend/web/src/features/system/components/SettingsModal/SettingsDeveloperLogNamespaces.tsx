import { Flex, FormControl, FormLabel, Tag, TagCloseButton, Text } from '@invoke-ai/ui-library';
import type { LogNamespace } from 'app/logging/logger';
import { zLogNamespace } from 'app/logging/logger';
import { useAppSelector } from 'app/store/storeHooks';
import { logNamespaceToggled, selectSystemLogNamespaces } from 'features/system/store/systemSlice';
import { difference } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useDispatch } from 'react-redux';

export const SettingsDeveloperLogNamespaces = memo(() => {
  const { t } = useTranslation();
  const enabledLogNamespaces = useAppSelector(selectSystemLogNamespaces);
  const disabledLogNamespaces = useMemo(
    () => difference(zLogNamespace.options, enabledLogNamespaces),
    [enabledLogNamespaces]
  );

  return (
    <FormControl orientation="vertical">
      <FormLabel>{t('system.logNamespaces.logNamespaces')}</FormLabel>
      <Flex w="full" gap={2} flexWrap="wrap" minH="32px" borderRadius="base" borderWidth={1} p={2} alignItems="center">
        {enabledLogNamespaces.map((namespace) => (
          <LogLevelTag key={`enabled-${namespace}`} namespace={namespace} isEnabled={true} />
        ))}
      </Flex>
      <Flex gap={2} flexWrap="wrap">
        {disabledLogNamespaces.map((namespace) => (
          <LogLevelTag key={`disabled-${namespace}`} namespace={namespace} isEnabled={false} />
        ))}
      </Flex>
    </FormControl>
  );
});

SettingsDeveloperLogNamespaces.displayName = 'SettingsDeveloperLogNamespaces';

const LogLevelTag = ({ namespace, isEnabled }: { namespace: LogNamespace; isEnabled: boolean }) => {
  const { t } = useTranslation();
  const dispatch = useDispatch();
  const onClick = useCallback(() => {
    dispatch(logNamespaceToggled(namespace));
  }, [dispatch, namespace]);

  return (
    <Tag
      h="min-content"
      borderRadius="base"
      onClick={onClick}
      colorScheme={isEnabled ? 'invokeBlue' : 'base'}
      userSelect="none"
      role="button"
      size="md"
      color="base.900"
      bg={isEnabled ? 'invokeBlue.300' : 'base.300'}
    >
      <Text fontSize="sm" fontWeight="semibold">
        {t(`system.logNamespaces.${namespace}`)}
      </Text>
      {isEnabled && <TagCloseButton />}
    </Tag>
  );
};
