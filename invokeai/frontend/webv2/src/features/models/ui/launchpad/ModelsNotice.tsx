import type { ModelBase, ModelConfig, StarterModelBundle } from '@features/models/core/types';

import { Flex, Icon, Link, Spinner, Stack, Text, Wrap } from '@chakra-ui/react';
import { ensureInstallsLoaded, isActiveInstallStatus, useInstallsSelector } from '@features/models/data/installsStore';
import { ensureModelsLoaded, useModelsSelector } from '@features/models/data/modelsStore';
import { ensureStartersLoaded, useStartersSelector } from '@features/models/data/startersStore';
import { getStarterBundleInstallSources } from '@features/models/ui/add-models/starterModelInstallSources';
import { useInstallActions } from '@features/models/ui/add-models/useInstallActions';
import { buildInstallQueueRows, summarizeInstallQueue } from '@features/models/ui/install-queue/queueModel';
import { useModelsUi } from '@features/models/ui/ModelsUiContext';
import { openAddModelsWithBundle, openInstallQueue } from '@features/models/ui/uiStore';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { useMountEffect } from '@platform/react/useMountEffect';
import { DOCS_URL } from '@platform/runtime/appMetadata';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { Button } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { useNavigate } from '@tanstack/react-router';
import { ArrowRightIcon, BoxIcon, DownloadIcon, ExternalLinkIcon, TriangleAlertIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState, type RefObject } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Home's first-run model guidance. Generation needs a main or hosted generator model; until one exists, managers get
 * a one-click install of the recommended starter bundle (and the others in the Model Manager), everyone else a
 * read-only notice naming who can add them. Managers also see installs in progress and installs that need attention.
 */

const EMPTY_BUNDLES: Record<string, StarterModelBundle> = {};

/**
 * First one the server offers wins: modest hardware needs and the broadest add-on support. Each has a "good for" line
 * (`models.launchpad.bundleGoodFor.<key>`); hardware figures live only in the docs' system requirements, linked here.
 */
const RECOMMENDED_BUNDLE_KEYS = ['sdxl', 'sd-1', 'z-image'] as const satisfies readonly ModelBase[];
const HARDWARE_REQUIREMENTS_URL = `${DOCS_URL}start-here/system-requirements/`;

const PANEL_STYLE = {
  bg: 'bg.subtle',
  borderColor: 'border.subtle',
  borderWidth: '1px',
  gap: '3',
  p: '4',
  rounded: 'lg',
} as const;

/** Models a generation can run on; encoders, adapters and upscalers alone cannot generate. */
const canGenerate = (model: ModelConfig): boolean => model.type === 'main' || model.type === 'external_image_generator';

export const ModelsNotice = () => {
  const { canManageModels } = useModelsUi();

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  const hasGenerator = useModelsSelector((snapshot) => snapshot.models.some(canGenerate));
  const modelsStatus = useModelsSelector((snapshot) => snapshot.status);
  // Wait for authoritative library data before claiming no models are installed.
  const isMissingModels = modelsStatus === 'loaded' && !hasGenerator;

  if (!canManageModels) {
    return isMissingModels ? <NoModelsReadOnly /> : null;
  }

  return <ManagerModelsNotice isMissingModels={isMissingModels} />;
};

/** Beside Home's "Start something" tiles: they still open the editor, but nothing generates without a model. */
export const ModelsRequiredHint = () => {
  const { t } = useTranslation();
  const { canManageModels } = useModelsUi();
  const isMissingModels = useModelsSelector(
    (snapshot) => snapshot.status === 'loaded' && !snapshot.models.some(canGenerate)
  );
  // The models panel loads installs for managers; nothing here asks the admin-only endpoint.
  const isInstalling = useInstallsSelector((snapshot) =>
    snapshot.jobs.some((job) => isActiveInstallStatus(job.status))
  );

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  if (!isMissingModels) {
    return null;
  }

  const messageKey = !canManageModels
    ? 'models.launchpad.startNeedsModelAskAdmin'
    : isInstalling
      ? 'models.launchpad.startAfterInstall'
      : 'models.launchpad.startNeedsModel';

  return (
    <Flex align="center" color="fg.muted" gap="1.5">
      <Icon as={TriangleAlertIcon} boxSize="3.5" color="fg.warning" />
      <Text fontSize="md">{t(messageKey)}</Text>
    </Flex>
  );
};

const NoModelsReadOnly = () => {
  const { t } = useTranslation();

  return (
    <Stack {...PANEL_STYLE} role="status">
      <NoticeTitle />
      <Text color="fg.muted" fontSize="md">
        {t('models.launchpad.noModelsAskAdmin')}
      </Text>
    </Stack>
  );
};

const NoticeTitle = ({ titleRef }: { titleRef?: RefObject<HTMLHeadingElement | null> }) => {
  const { t } = useTranslation();

  return (
    <Flex align="center" gap="2">
      <Icon as={BoxIcon} boxSize="4" color="fg.muted" />
      {/* Focusable by script only: it takes focus when the control the viewer used goes away. */}
      <Text as="h2" fontSize="lg" fontWeight="700" outline="none" ref={titleRef} tabIndex={-1}>
        {t('models.launchpad.noModelsTitle')}
      </Text>
    </Flex>
  );
};

/** Starters and installs are admin-only endpoints, so only this branch loads them. */
const ManagerModelsNotice = ({ isMissingModels }: { isMissingModels: boolean }) => {
  const { t } = useTranslation();
  const bundles = useStartersSelector((snapshot) => snapshot.response?.starter_bundles ?? EMPTY_BUNDLES);
  const jobs = useInstallsSelector((snapshot) => snapshot.jobs);
  const dismissedJobIds = useInstallsSelector((snapshot) => snapshot.dismissedJobIds);
  const activeInstallCount = useMemo(() => jobs.filter((job) => isActiveInstallStatus(job.status)).length, [jobs]);
  const attentionCount = useMemo(
    () => summarizeInstallQueue(buildInstallQueueRows(jobs, dismissedJobIds)).attention,
    [dismissedJobIds, jobs]
  );

  const titleRef = useRef<HTMLHeadingElement>(null);
  // Queuing swaps the bundles for install progress; focus stays in the panel instead of falling to the page.
  const focusTitle = useCallback(() => titleRef.current?.focus(), []);

  useMountEffect(() => {
    ensureStartersLoaded();
    ensureInstallsLoaded();
  });

  const activity =
    activeInstallCount > 0 || attentionCount > 0 ? (
      <Stack gap="2">
        {activeInstallCount > 0 ? <InstallProgress count={activeInstallCount} /> : null}
        {attentionCount > 0 ? <InstallsNeedAttention count={attentionCount} /> : null}
      </Stack>
    ) : null;

  if (!isMissingModels) {
    return activity ? <Stack {...PANEL_STYLE}>{activity}</Stack> : null;
  }

  return (
    <Stack {...PANEL_STYLE}>
      <NoticeTitle titleRef={titleRef} />
      <Text color="fg.muted" fontSize="md">
        {t('models.launchpad.noModelsDescription')}{' '}
        <Link color="fg" fontSize="md" href={HARDWARE_REQUIREMENTS_URL} rel="noreferrer" target="_blank">
          {t('models.launchpad.hardwareRequirements')}
          <ExternalLinkIcon size={12} />
        </Link>
      </Text>
      {activity}
      {activeInstallCount > 0 ? null : <StarterBundles bundles={bundles} onQueued={focusTitle} />}
    </Stack>
  );
};

/** Opens the Model Manager on its install queue. */
const useOpenInstallQueue = () => {
  const navigate = useNavigate();

  return useCallback(() => {
    openInstallQueue();
    void navigate({ to: '/models' });
  }, [navigate]);
};

const InstallProgress = ({ count }: { count: number }) => {
  const { t } = useTranslation();
  const openQueue = useOpenInstallQueue();

  return (
    <Flex align="center" gap="2" wrap="wrap">
      <Spinner color="fg.muted" />
      <Text color="fg.muted" fontSize="md" role="status">
        {t('models.launchpad.installing', { count })}
      </Text>
      <Button size="sm" variant="outline" onClick={openQueue}>
        {t('models.launchpad.openQueue')}
        <ArrowRightIcon />
      </Button>
    </Flex>
  );
};

const InstallsNeedAttention = ({ count }: { count: number }) => {
  const { t } = useTranslation();
  const review = useOpenInstallQueue();

  return (
    <Flex align="center" gap="2" wrap="wrap">
      <Icon as={TriangleAlertIcon} boxSize="4" color="fg.error" />
      <Text fontSize="md">{t('models.launchpad.installsNeedAttention', { count })}</Text>
      <Button size="sm" variant="outline" onClick={review}>
        {t('models.launchpad.reviewInstalls')}
        <ArrowRightIcon />
      </Button>
    </Flex>
  );
};

const StarterBundles = ({
  bundles,
  onQueued,
}: {
  bundles: Record<string, StarterModelBundle>;
  onQueued: () => void;
}) => {
  const { t } = useTranslation();
  const entries = useMemo(() => Object.entries(bundles), [bundles]);
  const recommendedKey = RECOMMENDED_BUNDLE_KEYS.find((key) => bundles[key] !== undefined) ?? null;
  const others = entries.filter(([key]) => key !== recommendedKey);

  if (entries.length === 0) {
    return null;
  }

  return (
    <Stack gap="3">
      {recommendedKey ? (
        <RecommendedBundle bundle={bundles[recommendedKey]!} bundleKey={recommendedKey} onQueued={onQueued} />
      ) : null}
      {others.length > 0 ? (
        <Stack gap="2">
          <Text color="fg.muted" fontSize="xs" fontWeight="600" textTransform="uppercase">
            {recommendedKey ? t('models.launchpad.otherBundles') : t('models.launchpad.starterBundles')}
          </Text>
          <Wrap gap="2">
            {others.map(([key, bundle]) => (
              <StarterBundleButton bundle={bundle} bundleKey={key} key={key} />
            ))}
          </Wrap>
        </Stack>
      ) : null}
    </Stack>
  );
};

/** One click queues every model of the bundle that is not installed yet, as Add Models' "Install bundle" does. */
const RecommendedBundle = ({
  bundle,
  bundleKey,
  onQueued,
}: {
  bundle: StarterModelBundle;
  bundleKey: string;
  onQueued: () => void;
}) => {
  const { i18n, t } = useTranslation();
  const notify = useNotify();
  const { installMany } = useInstallActions();
  const [isInstalling, setIsInstalling] = useState(false);
  const name = bundle.name || bundleKey;
  const goodForKey = `models.launchpad.bundleGoodFor.${bundleKey}`;

  const install = useCallback(async () => {
    const owner = captureAccountScope();

    setIsInstalling(true);

    try {
      const queued = await installMany(getStarterBundleInstallSources(bundle));

      if (queued > 0 && isAccountScopeCurrent(owner)) {
        notify.success(
          t('models.bundleInstallQueued'),
          t('models.bundleInstallQueuedDescription', { count: queued, name })
        );
        onQueued();
      }
    } finally {
      if (isAccountScopeCurrent(owner)) {
        setIsInstalling(false);
      }
    }
  }, [bundle, installMany, name, notify, onQueued, t]);
  const handleInstall = useCallback(() => void install(), [install]);

  return (
    <Flex
      align="center"
      bg="bg"
      borderColor="border.subtle"
      borderWidth="1px"
      gap="3"
      justify="space-between"
      p="3"
      rounded="md"
      wrap="wrap"
    >
      <Stack gap="0.5" minW="0">
        <Text color="fg.muted" fontSize="xs" fontWeight="600" textTransform="uppercase">
          {t('models.launchpad.recommended')}
        </Text>
        <Text fontSize="md" fontWeight="600">
          {name} · {t('models.modelCount', { count: bundle.models.length })}
        </Text>
        {i18n.exists(goodForKey) ? (
          <Text color="fg.muted" fontSize="md">
            {t(goodForKey)}
          </Text>
        ) : null}
      </Stack>
      <Button flexShrink={0} loading={isInstalling} variant="solid" onClick={handleInstall}>
        <DownloadIcon />
        {t('models.installNamedBundle', { name })}
      </Button>
    </Flex>
  );
};

const StarterBundleButton = ({ bundle, bundleKey }: { bundle: StarterModelBundle; bundleKey: string }) => {
  const { i18n, t } = useTranslation();
  const navigate = useNavigate();
  const goodForKey = `models.launchpad.bundleGoodFor.${bundleKey}`;

  const handleOpen = useCallback(() => {
    openAddModelsWithBundle(bundle.name || bundleKey);
    void navigate({ to: '/models' });
  }, [bundle.name, bundleKey, navigate]);

  const button = (
    <Button variant="outline" onClick={handleOpen}>
      {bundle.name || bundleKey}
      <Icon as={ArrowRightIcon} boxSize="3" color="fg.muted" />
    </Button>
  );

  return i18n.exists(goodForKey) ? <Tooltip content={t(goodForKey)}>{button}</Tooltip> : button;
};
