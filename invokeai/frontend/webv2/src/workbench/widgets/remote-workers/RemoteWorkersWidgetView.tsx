import type { WidgetViewProps } from '@workbench/widgetContracts';
import type { ChangeEvent } from 'react';

import { Badge, Box, Button, HStack, Input, NativeSelect, Stack, Switch, Text, Textarea } from '@chakra-ui/react';
import {
  getRemoteWorkerUrls,
  invalidateRemoteWorkerHealth,
  refreshRemoteWorkerHealth,
  remoteWorkersHealthStore,
  remoteWorkersStore,
  setRemoteWorkerEnabled,
  setRemoteWorkerName,
  setRemoteWorkersSettings,
  type RemoteDispatchMode,
} from '@features/queue';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { apiFetchJson, getApiErrorMessage } from '@platform/transport/http';
import { ChevronDownIcon, ChevronUpIcon, PencilIcon, PowerIcon } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

const handleEnabledChange = (details: { checked: boolean }): void => {
  setRemoteWorkersSettings({ enabled: details.checked });
};

const handleDispatchModeChange = (event: ChangeEvent<HTMLSelectElement>): void => {
  setRemoteWorkersSettings({ dispatchMode: event.target.value as RemoteDispatchMode });
};

const handleWorkerUrlsChange = (event: ChangeEvent<HTMLTextAreaElement>): void => {
  setRemoteWorkersSettings({ workerUrls: event.target.value });
};

const handleAutoTransferChange = (details: { checked: boolean }): void => {
  setRemoteWorkersSettings({ autoTransferMissingModels: details.checked });
};

const handleKeepCopiesChange = (details: { checked: boolean }): void => {
  setRemoteWorkersSettings({ keepRemoteCopies: details.checked });
};

const handleTransferHostChange = (event: ChangeEvent<HTMLInputElement>): void => {
  setRemoteWorkersSettings({ modelTransferHost: event.target.value });
};

interface CredentialStatus {
  saved: boolean;
  email: string | null;
}

/** The password never enters queue settings, localStorage, or the workflow graph. */
const WorkerAuthRow = ({ enabled, slot, url }: { enabled: boolean; slot: number; url: string }) => {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useState(false);
  const workerEnabled = remoteWorkersStore.useSelector(
    (settings) => !settings.disabledWorkerUrls.includes(url.toLowerCase())
  );
  const savedName = remoteWorkersStore.useSelector((settings) => settings.workerNames[url.toLowerCase()]?.trim() ?? '');
  const defaultName = t('widgets.remoteWorkers.worker.defaultName', { slot });
  const name = savedName || defaultName;
  const availability = remoteWorkersHealthStore.useSnapshot().byUrl[url]?.status ?? 'checking';
  const availabilityLabel = !workerEnabled
    ? t('widgets.remoteWorkers.status.disabled')
    : !enabled
      ? t('widgets.remoteWorkers.status.paused')
      : availability === 'online'
        ? t('widgets.remoteWorkers.status.online')
        : availability === 'offline'
          ? t('widgets.remoteWorkers.status.offline')
          : availability === 'login_required'
            ? t('widgets.remoteWorkers.status.loginRequired')
            : t('widgets.remoteWorkers.status.checking');
  const availabilityColor =
    !workerEnabled || !enabled
      ? 'gray'
      : availability === 'online'
        ? 'green'
        : availability === 'offline'
          ? 'red'
          : availability === 'login_required'
            ? 'orange'
            : 'gray';
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [saved, setSaved] = useState(false);
  const [checkingLogin, setCheckingLogin] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => {
    let active = true;
    void apiFetchJson<CredentialStatus>(`/api/v1/remote_workers/credentials?url=${encodeURIComponent(url)}`)
      .then((status) => {
        if (!active) {
          return;
        }
        setSaved(status.saved);
        setEmail(status.email ?? '');
        setCheckingLogin(false);
        setMessage('');
      })
      .catch((error: unknown) => {
        if (active) {
          setCheckingLogin(false);
          setMessage(getApiErrorMessage(error, t('widgets.remoteWorkers.login.loadError')));
        }
      });
    return () => {
      active = false;
    };
  }, [t, url]);

  const handleEmailChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setEmail(event.target.value);
  }, []);
  const handlePasswordChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setPassword(event.target.value);
  }, []);
  const handleNameBlur = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const value = event.currentTarget.value;
      setRemoteWorkerName(url, value);
      if (!value.trim()) {
        event.currentTarget.value = defaultName;
      }
    },
    [defaultName, url]
  );
  const refreshHealthAfterCredentialChange = useCallback(() => {
    invalidateRemoteWorkerHealth(url);
    if (enabled) {
      void refreshRemoteWorkerHealth([url]);
    }
  }, [enabled, url]);
  const handleSave = useCallback(async () => {
    setBusy(true);
    setMessage('');
    try {
      const status = await apiFetchJson<CredentialStatus>('/api/v1/remote_workers/credentials', {
        method: 'PUT',
        body: JSON.stringify({ url, email, password, remember_me: true }),
      });
      setSaved(status.saved);
      setEmail(status.email ?? '');
      setPassword('');
      refreshHealthAfterCredentialChange();
      setMessage(t('widgets.remoteWorkers.login.savedMessage'));
    } catch (error) {
      setMessage(getApiErrorMessage(error, t('widgets.remoteWorkers.login.saveError')));
    } finally {
      setBusy(false);
    }
  }, [url, email, password, refreshHealthAfterCredentialChange, t]);
  const handleRemove = useCallback(async () => {
    setBusy(true);
    setMessage('');
    try {
      await apiFetchJson<CredentialStatus>(`/api/v1/remote_workers/credentials?url=${encodeURIComponent(url)}`, {
        method: 'DELETE',
      });
      setSaved(false);
      setEmail('');
      setPassword('');
      refreshHealthAfterCredentialChange();
      setMessage(t('widgets.remoteWorkers.login.removedMessage'));
    } catch (error) {
      setMessage(getApiErrorMessage(error, t('widgets.remoteWorkers.login.removeError')));
    } finally {
      setBusy(false);
    }
  }, [url, refreshHealthAfterCredentialChange, t]);

  const toggleExpanded = useCallback(() => setExpanded((open) => !open), []);
  const toggleWorkerEnabled = useCallback(() => setRemoteWorkerEnabled(url, !workerEnabled), [url, workerEnabled]);

  return (
    <Box borderBottomColor="border.subtle" borderBottomWidth="1px" pb="2" pt="1">
      <HStack align="center" gap="1">
        <Button
          aria-expanded={expanded}
          aria-label={t('widgets.remoteWorkers.worker.settings', { slot })}
          color="fg"
          flex="1"
          h="auto"
          justifyContent="space-between"
          minW="0"
          onClick={toggleExpanded}
          px="1"
          py="2"
          size="sm"
          type="button"
          variant="ghost"
        >
          <HStack align="center" flex="1" gap="2" minW="0" textAlign="start">
            <Badge colorPalette={workerEnabled ? availabilityColor : 'gray'} flexShrink={0} variant="subtle">
              R{slot}
            </Badge>
            <Stack flex="1" gap="0" minW="0" opacity={workerEnabled ? 1 : 0.55}>
              <Text fontSize="sm" fontWeight="medium">
                {name}
              </Text>
              <Text color="fg.muted" fontFamily="mono" fontSize="2xs" overflowWrap="anywhere" whiteSpace="normal">
                {url}
              </Text>
            </Stack>
          </HStack>
          <Badge colorPalette={availabilityColor} flexShrink={0} variant="subtle">
            {availabilityLabel}
          </Badge>
        </Button>
        <Button
          aria-label={t(
            workerEnabled
              ? 'widgets.remoteWorkers.worker.disableForNewJobs'
              : 'widgets.remoteWorkers.worker.enableForNewJobs',
            { slot }
          )}
          aria-pressed={workerEnabled}
          color={workerEnabled ? 'green.400' : 'fg.muted'}
          h="7"
          minW="7"
          onClick={toggleWorkerEnabled}
          px="0"
          size="xs"
          title={t(
            workerEnabled
              ? 'widgets.remoteWorkers.worker.disableForNewJobs'
              : 'widgets.remoteWorkers.worker.enableForNewJobs',
            { slot }
          )}
          type="button"
          variant="ghost"
        >
          <PowerIcon size={15} />
        </Button>
        <Button
          aria-expanded={expanded}
          aria-label={t(
            expanded ? 'widgets.remoteWorkers.worker.collapseSettings' : 'widgets.remoteWorkers.worker.expandSettings',
            { slot }
          )}
          color="fg.muted"
          h="7"
          minW="6"
          onClick={toggleExpanded}
          px="0"
          size="xs"
          type="button"
          variant="ghost"
        >
          {expanded ? <ChevronUpIcon size={14} /> : <ChevronDownIcon size={14} />}
        </Button>
      </HStack>
      {expanded ? (
        <Stack gap="2" pb="2" pt="2" px="1">
          <Input
            aria-label={t('widgets.remoteWorkers.worker.nameLabel', { slot })}
            key={`${url}:${savedName}:${defaultName}`}
            defaultValue={name}
            onBlur={handleNameBlur}
            placeholder={defaultName}
            size="sm"
          />
          <Badge alignSelf="start" colorPalette={saved ? 'green' : 'gray'} variant="subtle">
            {checkingLogin
              ? t('widgets.remoteWorkers.login.checking')
              : saved
                ? t('widgets.remoteWorkers.login.saved')
                : t('widgets.remoteWorkers.login.none')}
          </Badge>
          <Text color="fg.muted" fontSize="xs">
            {t('widgets.remoteWorkers.login.description')}
          </Text>
          <Input
            autoComplete="off"
            onChange={handleEmailChange}
            placeholder={t('widgets.remoteWorkers.login.emailPlaceholder')}
            size="sm"
            type="email"
            value={email}
          />
          <Input
            autoComplete="new-password"
            onChange={handlePasswordChange}
            placeholder={
              saved
                ? t('widgets.remoteWorkers.login.replacePasswordPlaceholder')
                : t('widgets.remoteWorkers.login.passwordPlaceholder')
            }
            size="sm"
            type="password"
            value={password}
          />
          <HStack gap="2">
            <Button disabled={busy || !email.trim() || !password} onClick={handleSave} size="sm">
              {t('widgets.remoteWorkers.login.save')}
            </Button>
            <Button disabled={busy || !saved} onClick={handleRemove} size="sm" variant="outline">
              {t('widgets.remoteWorkers.login.remove')}
            </Button>
          </HStack>
        </Stack>
      ) : null}
      {message ? (
        <Text color="fg.muted" fontSize="xs" px="1">
          {message}
        </Text>
      ) : null}
    </Box>
  );
};

/** Availability polling is display-only; backend workers decide job eligibility. */
export const RemoteWorkersWidgetView = (_props: WidgetViewProps) => {
  const { t } = useTranslation();
  const settings = remoteWorkersStore.useSnapshot();
  const urls = useMemo(() => getRemoteWorkerUrls(settings.workerUrls), [settings.workerUrls]);
  const accountId = captureAccountScope().accountId;
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showWorkerEditor, setShowWorkerEditor] = useState(false);

  useEffect(() => {
    if (!settings.enabled) {
      if (Object.keys(remoteWorkersHealthStore.getSnapshot().byUrl).length > 0) {
        remoteWorkersHealthStore.setSnapshot({ byUrl: {} });
      }
      return;
    }

    const refresh = (): void => {
      void refreshRemoteWorkerHealth(urls);
    };

    refresh();
    const timer = globalThis.setInterval(refresh, 15_000);
    return () => globalThis.clearInterval(timer);
  }, [settings.enabled, urls]);
  const handleAdvancedChange = useCallback((details: { checked: boolean }) => {
    setShowAdvanced(details.checked);
  }, []);
  const toggleWorkerEditor = useCallback(() => setShowWorkerEditor((open) => !open), []);

  return (
    <Stack gap="5" p="3">
      <Stack gap="3">
        <HStack justify="space-between">
          <Text fontWeight="semibold">{t('widgets.remoteWorkers.title')}</Text>
          <Badge colorPalette={settings.enabled ? 'green' : 'gray'}>
            {settings.enabled ? t('widgets.remoteWorkers.enabled') : t('widgets.remoteWorkers.disabled')}
          </Badge>
        </HStack>
        <Switch.Root checked={settings.enabled} onCheckedChange={handleEnabledChange}>
          <Switch.HiddenInput />
          <Switch.Control>
            <Switch.Thumb />
          </Switch.Control>
          <Switch.Label>{t('widgets.remoteWorkers.enable')}</Switch.Label>
        </Switch.Root>
        <Text color="fg.muted" fontSize="xs">
          {t('widgets.remoteWorkers.description')}
        </Text>
      </Stack>

      <Box borderColor="border.subtle" borderTopWidth="1px" pt="4">
        <Stack gap="2">
          <Text fontSize="sm" fontWeight="semibold">
            {t('widgets.remoteWorkers.dispatch.title')}
          </Text>
          <NativeSelect.Root size="sm" disabled={!settings.enabled}>
            <NativeSelect.Field value={settings.dispatchMode} onChange={handleDispatchModeChange}>
              <option value="distributed">{t('widgets.remoteWorkers.dispatch.distributed')}</option>
              <option value="remote_only">{t('widgets.remoteWorkers.dispatch.remoteOnly')}</option>
            </NativeSelect.Field>
            <NativeSelect.Indicator />
          </NativeSelect.Root>
          <Text color="fg.muted" fontSize="xs">
            {settings.dispatchMode === 'remote_only'
              ? t('widgets.remoteWorkers.dispatch.remoteOnlyDescription')
              : t('widgets.remoteWorkers.dispatch.distributedDescription')}
          </Text>
        </Stack>
      </Box>

      <Box borderColor="border.subtle" borderTopWidth="1px" pt="4">
        <Stack gap="2">
          <HStack justify="space-between" gap="2">
            <Text fontSize="sm" fontWeight="semibold">
              {t('widgets.remoteWorkers.workers.title')}
            </Text>
            <Badge colorPalette="gray" variant="subtle">
              {t('widgets.remoteWorkers.workers.configured', { count: urls.length })}
            </Badge>
          </HStack>
          {urls.length === 0 ? (
            <Text color={settings.enabled ? 'fg.warning' : 'fg.muted'} fontSize="xs">
              {t('widgets.remoteWorkers.workers.none')}
            </Text>
          ) : (
            <Stack gap="1">
              {urls.map((url, index) => (
                <WorkerAuthRow enabled={settings.enabled} key={`${accountId}:${url}`} slot={index + 1} url={url} />
              ))}
            </Stack>
          )}
          <Button alignSelf="start" onClick={toggleWorkerEditor} size="xs" variant="outline">
            <PencilIcon size={13} />
            {showWorkerEditor
              ? t('widgets.remoteWorkers.workers.doneEditingAddresses')
              : t('widgets.remoteWorkers.workers.editAddresses')}
          </Button>
          {showWorkerEditor ? (
            <Stack gap="1">
              <Text color="fg.muted" fontSize="xs">
                {t('widgets.remoteWorkers.workers.editorDescription')}
              </Text>
              <Textarea
                aria-label={t('widgets.remoteWorkers.workers.urlsLabel')}
                fontFamily="mono"
                fontSize="sm"
                onChange={handleWorkerUrlsChange}
                placeholder={'http://192.168.1.100:9090\nhttp://192.168.1.101:9090'}
                resize="vertical"
                rows={3}
                value={settings.workerUrls}
              />
            </Stack>
          ) : null}
          <Text color="fg.muted" fontSize="xs">
            {t('widgets.remoteWorkers.workers.statusHelp')}
          </Text>
        </Stack>
      </Box>

      <Box borderColor="border.subtle" borderTopWidth="1px" pt="4">
        <Stack gap="3">
          <Text fontSize="sm" fontWeight="semibold">
            {t('widgets.remoteWorkers.modelTransfer.title')}
          </Text>
          <Switch.Root checked={settings.autoTransferMissingModels} onCheckedChange={handleAutoTransferChange}>
            <Switch.HiddenInput />
            <Switch.Control>
              <Switch.Thumb />
            </Switch.Control>
            <Switch.Label>{t('widgets.remoteWorkers.modelTransfer.transferMissing')}</Switch.Label>
          </Switch.Root>
          <Switch.Root checked={settings.keepRemoteCopies} onCheckedChange={handleKeepCopiesChange}>
            <Switch.HiddenInput />
            <Switch.Control>
              <Switch.Thumb />
            </Switch.Control>
            <Switch.Label>{t('widgets.remoteWorkers.modelTransfer.keepCopies')}</Switch.Label>
          </Switch.Root>
          {settings.autoTransferMissingModels ? (
            <>
              <Switch.Root checked={showAdvanced} onCheckedChange={handleAdvancedChange}>
                <Switch.HiddenInput />
                <Switch.Control>
                  <Switch.Thumb />
                </Switch.Control>
                <Switch.Label>{t('widgets.remoteWorkers.modelTransfer.advanced')}</Switch.Label>
              </Switch.Root>
              {showAdvanced ? (
                <Stack gap="1">
                  <Text fontSize="sm" fontWeight="medium">
                    {t('widgets.remoteWorkers.modelTransfer.hostLabel')}
                  </Text>
                  <Input
                    fontFamily="mono"
                    onChange={handleTransferHostChange}
                    placeholder={t('widgets.remoteWorkers.modelTransfer.hostPlaceholder')}
                    size="sm"
                    value={settings.modelTransferHost}
                  />
                </Stack>
              ) : null}
            </>
          ) : null}
        </Stack>
      </Box>

      <Box borderColor="border.subtle" borderTopWidth="1px" pt="4">
        <Text color="fg.muted" fontSize="xs">
          {t('widgets.remoteWorkers.destinationHelp')}
        </Text>
      </Box>
    </Stack>
  );
};
